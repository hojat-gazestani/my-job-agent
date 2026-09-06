from typing import Any, TypedDict


class AgentState(TypedDict):
    """
    Langgraph shared state
    """

    queries: list[str]
    raw_results: list[dict[str, Any]]
    verified_jobs: list[dict[str, Any]]
    final_report: str
