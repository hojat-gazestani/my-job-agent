import json
from pathlib import Path

import networkx as nx

from models import KnowledgeGraph


def extract_knowledge(llm, conversation_text: str) -> KnowledgeGraph:
    prompt = f"""
    Extract entities and relationships from this conversation about job applications.
    Return nodes for: search session, jobs, companies, applied jobs.
    Return edges for: searched, belongs_to, applied relationships.

    Conversation:
    {conversation_text}
    """
    structured_llm = llm.with_structured_output(KnowledgeGraph)
    return structured_llm.invoke(prompt)


def update_graph(graph: nx.DiGraph, kg: KnowledgeGraph) -> nx.DiGraph:
    for node in kg.nodes:
        if node.id not in graph:
            graph.add_node(node.id, type=node.type, **node.properties)
    for edge in kg.edges:
        if not graph.has_edge(edge.source, edge.target):
            graph.add_edge(edge.source, edge.target, label=edge.label)
    return graph


def save_graph_to_disk(graph: nx.DiGraph, path: Path):
    data = nx.node_link_data(graph)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def load_graph_from_disk(path: Path) -> nx.DiGraph:
    if not path.exists():
        return nx.DiGraph()
    with path.open("r") as f:
        data = json.load(f)
    return nx.node_link_graph(data, directed=True)
