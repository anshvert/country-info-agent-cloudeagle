import os
import json
from functools import lru_cache
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

from .state import AgentState
from .tools import fetch_country

load_dotenv()


@lru_cache(maxsize=1)
def _get_llm() -> ChatOpenRouter:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set in environment.")
    return ChatOpenRouter(model="x-ai/grok-4.1-fast", api_key=api_key)


_KNOWN_FIELDS = [
    "name", "population", "capital", "currency", "language",
    "region", "subregion", "area", "flag", "timezone",
    "continent", "borders", "calling code",
]

_INTENT_SYSTEM = f"""You are an intent parser for a country information service.

Given a user question (and optional prior conversation context), extract:
1. The country name (as it would be searched in English)
2. Which fields they want to know about

Return ONLY valid JSON in this exact format:
{{"country": "<country name or null>", "fields": ["<field1>", "<field2>"]}}

Available fields: {", ".join(_KNOWN_FIELDS)}

Rules:
- If no country is mentioned but context implies one, infer it from the conversation
- If no country can be determined at all, set country to null
- If they ask "everything" or "all info", include all relevant fields
- Return ONLY the JSON object, nothing else"""

_SYNTHESIS_SYSTEM = """You are a concise, factual assistant answering questions about countries.

You will receive:
- The user's original question
- Raw country data from the REST Countries API

Rules:
- Answer directly and factually using ONLY the provided data
- If a specific field is missing from the data, say "that information is not available"
- Keep answers concise but complete
- Format numbers with commas (e.g., 83,000,000)
- Do not add any information not present in the raw data"""


def _format_history(history: list[dict]) -> str:
    if not history:
        return ""
    lines = ["Prior conversation context:"]
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines) + "\n\n"


async def extract_intent(state: AgentState) -> AgentState:
    history_context = _format_history(state.get("history", []))
    user_input = f"{history_context}Current question: {state['question']}"

    response = await _get_llm().ainvoke([
        SystemMessage(content=_INTENT_SYSTEM),
        HumanMessage(content=user_input),
    ])
    try:
        parsed = json.loads(response.content.strip())
        return {
            **state,
            "country": parsed.get("country"),
            "fields": parsed.get("fields", []),
            "error": None,
        }
    except (json.JSONDecodeError, AttributeError):
        return {
            **state,
            "country": None,
            "fields": [],
            "error": "Could not parse intent from question.",
        }


async def fetch_data(state: AgentState) -> AgentState:
    if state.get("error"):
        return state

    country = state.get("country")
    if not country:
        return {
            **state,
            "raw_data": None,
            "error": "I couldn't identify a country in your question. Please ask about a specific country.",
        }

    try:
        data = await fetch_country(country)
        return {**state, "raw_data": data, "error": None}
    except ValueError as e:
        return {**state, "raw_data": None, "error": str(e)}
    except Exception:
        return {
            **state,
            "raw_data": None,
            "error": "Failed to fetch country data. Please try again.",
        }


async def synthesize_answer(state: AgentState) -> AgentState:
    if state.get("error"):
        return {**state, "answer": state["error"]}

    raw_data = state.get("raw_data", {})
    fields = state.get("fields", [])
    question = state["question"]

    data_summary = json.dumps(raw_data, ensure_ascii=False, indent=2)

    prompt = f"""User question: {question}

Requested fields: {", ".join(fields) if fields else "general info"}

Raw country data:
{data_summary}"""

    response = await _get_llm().ainvoke([
        SystemMessage(content=_SYNTHESIS_SYSTEM),
        HumanMessage(content=prompt),
    ])

    return {**state, "answer": response.content.strip()}
