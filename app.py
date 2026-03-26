import chainlit as cl

from agent.graph import agent
from agent.state import AgentState

_MAX_HISTORY = 6 


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(
        content=(
            "👋 **Country Info Agent** ready!\n\n"
            "Ask me anything about any country — population, capital, currency, languages, and more.\n\n"
            "**Examples:**\n"
            "- What is the population of Germany?\n"
            "- What currency does Japan use?\n"
            "- What is the capital and population of Brazil?\n"
            "- Follow up: What are its official languages?"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    question = message.content.strip()
    if not question:
        return

    history: list[dict] = cl.user_session.get("history", [])

    initial_state = AgentState(
        question=question,
        history=history[-_MAX_HISTORY:],
        country=None,
        fields=[],
        raw_data=None,
        answer=None,
        error=None,
    )

    current_state: dict = dict(initial_state)

    async for chunk in agent.astream(initial_state, stream_mode="updates"):
        for node_name, node_update in chunk.items():
            current_state.update(node_update)

            if node_name == "extract_intent":
                country = node_update.get("country") or "unknown"
                fields = node_update.get("fields", [])
                async with cl.Step(name="🧠 Intent extracted") as step:
                    step.output = f"Country: **{country}** | Fields: {', '.join(fields) or 'general info'}"

            elif node_name == "fetch_data":
                has_data = bool(node_update.get("raw_data"))
                async with cl.Step(name="🌐 Country data fetched") as step:
                    step.output = "Data retrieved from REST Countries API." if has_data else "No data available."

            elif node_name == "synthesize_answer":
                async with cl.Step(name="✍️ Answer composed") as step:
                    step.output = "Response generated from live data."

    answer = (
        current_state.get("answer")
        or current_state.get("error")
        or "Sorry, I couldn't process that request."
    )

    await cl.Message(content=answer).send()

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    cl.user_session.set("history", history)
