from langgraph.graph import END, START, StateGraph

from agents.nodes import (
    planner_node,
    scorer_node,
    search_node,
    verifier_node,
)
from agents.state import AgentState


def build_graph():
    # Langgraph Pipeline

    builder = StateGraph(AgentState)

    builder.add_node("planner", planner_node)
    builder.add_node("searcher", search_node)
    builder.add_node("verifier", verifier_node)
    builder.add_node("scorer", scorer_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "searcher")
    builder.add_edge("searcher", "verifier")
    builder.add_edge("verifier", "scorer")
    builder.add_edge("scorer", END)

    graph = builder.compile()
    return graph
