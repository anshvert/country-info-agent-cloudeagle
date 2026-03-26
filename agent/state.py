from typing import TypedDict, Optional


class AgentState(TypedDict):
    question: str
    history: list[dict]
    country: Optional[str]
    fields: list[str]
    raw_data: Optional[dict]
    answer: Optional[str]
    error: Optional[str]
