from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    id: str
    type: str  # "search_session", "job", "company", "applied_job"
    properties: dict = Field(default_factory=dict)


class GraphEdge(BaseModel):
    source: str
    target: str
    label: str  # "searched", "belongs_to", "applied"


class KnowledgeGraph(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
