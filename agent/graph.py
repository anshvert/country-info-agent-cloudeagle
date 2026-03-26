from langgraph.graph import StateGraph, END

from .state import AgentState
from .nodes import extract_intent, fetch_data, synthesize_answer


def _should_continue(state: AgentState) -> str:
    if state.get("error") and not state.get("raw_data"):
        return "synthesize"
    return "fetch"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("extract_intent", extract_intent)
    graph.add_node("fetch_data", fetch_data)
    graph.add_node("synthesize_answer", synthesize_answer)

    graph.set_entry_point("extract_intent")

    graph.add_conditional_edges(
        "extract_intent",
        _should_continue,
        {
            "fetch": "fetch_data",
            "synthesize": "synthesize_answer",
        },
    )

    graph.add_edge("fetch_data", "synthesize_answer")
    graph.add_edge("synthesize_answer", END)

    return graph.compile()


agent = build_graph()
