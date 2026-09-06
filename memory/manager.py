import uuid
from pathlib import Path

from .graph import (
    load_graph_from_disk,
    save_graph_to_disk,
)

GRAPH_PATH = Path(__file__).parent.parent / "graph_memory.json"


class MemoryManager:
    def __init__(self, llm):
        self.llm = llm
        self.graph = load_graph_from_disk(GRAPH_PATH)

    def save_graph(self) -> None:
        save_graph_to_disk(self.graph, GRAPH_PATH)

    def get_applied_jobs(self) -> list[dict]:
        # Return nodes where node["type"] == "applied_job"
        return [
            {"id": node, **data}
            for node, data in self.graph.nodes(data=True)
            if data.get("type") == "applied_job"
        ]

    def get_applied_companies(self) -> set[str]:
        # Get company names from applied_job nodes
        companies = set()
        for node, data in self.graph.nodes(data=True):
            if data.get("type") == "applied_job":
                company = data.get("company")
                if company:
                    companies.add(company)
        return companies

    def mark_applied(self, session_id: str, jobs: list[dict]) -> None:
        # For each job, add as "applied_job" node + edges
        for job in jobs:
            job_id = str(job.get("url", ""))
            self.graph.add_node(
                job_id,
                type="applied_job",
                title=job.get("title"),
                company=job.get("company"),
                location=job.get("location"),
                applied_date=str(uuid.uuid4())[:8],
            )
            # Edge: session -> applied job
            session_node = f"session:{session_id}"
            if session_node in self.graph:
                self.graph.add_edge(session_node, job_id, label="applied")
        self.save_graph()

    def add_search_session(self, session_id: str, query_summary: str) -> None:
        session_node = f"session:{session_id}"
        self.graph.add_node(
            session_node,
            type="search_session",
            query_summary=query_summary,
        )
        self.save_graph()
