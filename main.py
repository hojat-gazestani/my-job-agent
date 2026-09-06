import json
import os
from pathlib import Path
from typing import Any, TypedDict

import requests
import yaml
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown
from tavily import TavilyClient

from models.models import JobReport

load_dotenv()
VLLM_KEY = os.getenv("OPENAI_API_KEY")
VLLM_HOST = os.getenv("VLLM_HOST")
VLLM_PORT = os.getenv("VLLM_PORT")
VLLM_MODEL = os.getenv("VLLM_MODEL")

existing_no_proxy = os.environ.get("NO_PROXY", "")
os.environ["NO_PROXY"] = f"{VLLM_HOST},{existing_no_proxy}".strip(",")

PROXY = os.getenv("HTTPS_PROXY")

session = requests.Session()
session.proxies = {"http": PROXY, "https": PROXY}
session.verify = "/etc/ssl/certs/ca-certificates.crt"

tavily = TavilyClient(
    api_key=os.environ["TAVILY_API_KEY"],
    session=session,
)

llm = ChatOpenAI(
    model=VLLM_MODEL,
    base_url=f"http://{VLLM_HOST}:{VLLM_PORT}/v1",
    api_key=VLLM_KEY,
)


# Load Candidate profile
PROFILE_PATH = Path(__file__).parent / "candidate_profile.yaml"
if not PROFILE_PATH.exists():
    raise FileNotFoundError(f"Profile not found: {PROFILE_PATH}")

with PROFILE_PATH.open("r", encoding="utf-8") as f:
    candidate_profile = yaml.safe_load(f)

if not isinstance(candidate_profile, dict):
    raise ValueError(f"Invalid candidate profile: {PROFILE_PATH}")


class AgentState(TypedDict):
    """
    Langgraph shared state
    """

    queries: list[str]
    raw_results: list[dict[str, Any]]
    verified_jobs: list[dict[str, Any]]
    final_report: str


class GeneratedQueries(BaseModel):
    """
    Pydantic Schema for Planner
    """

    queries: list[str] = Field(
        description="List of targeted queries derived from candidate profile."
    )


def render_markdown(report: JobReport) -> str:
    lines = [
        "# Job Matching Evaluation Report",
        "",
        "## Top matches",
        "",
        "| Job Title | Company | Location | Fit score | Visa Status | Link |",
        "|---|---|---|---|---|---|",
    ]

    for job in sorted(report.jobs, key=lambda x: x.fit_score, reverse=True):
        lines.append(
            f"| {job.title} | {job.company} | {job.location} | "
            f"{job.fit_score}% | {job.visa_status} |"
            f"[View Job]({job.url}) |"
        )

    lines.extend(["", "## Detailed Breakdown", ""])

    for i, job in enumerate(
        sorted(report.jobs, key=lambda x: x.fit_score, reverse=True),
        1,
    ):
        lines.extend(
            [
                f"### {i}. {job.title}",
                f"- **Company:** {job.company}",
                f"- **Location:** {job.location}",
                f"- **Fit Score:** {job.fit_score}",
                f"- **Visa Status:** {job.visa_status}",
                f"- **Reason:** {job.reason}",
                "",
            ]
        )
    return "\n".join(lines)


# Node 1: PLanner Agent
def planner_node(state: AgentState) -> dict[str, Any]:
    print("\n[Planner Agent] Generating search queries from candidate profile ...")

    planner_llm = llm.with_structured_output(GeneratedQueries)

    prompt = f"""
    Analyze this candidate profile:
    {json.dumps(candidate_profile, indent=2)}

    Generate 6 targeted search queries to find current job listings in Europe
    with visa sponsorship. Derive specific keyword combinations using ATS
    domains like site:greenhouse.io or site:lever.co combined with stack variants.

    Rules:
    - Target diverse roles that match their experience (e.g., Platform, SRE,
      Developer Productivity, Infrastructure).
    """

    res = planner_llm.invoke(
        [
            SystemMessage(content="You are an expert Executive Tech Recruiter."),
            HumanMessage(content=prompt),
        ]
    )
    print(f"Generated {len(res.queries)} targeted queries.")
    return {"queries": res.queries}


# Node 2: Search Tool Executor
def search_node(state: AgentState) -> dict[str, Any]:
    print("\n[Search Tool] Executing Tavily searches...")
    raw_results = []
    seen_urls = set()

    for query_text in state["queries"]:
        print(f" Searching: {query_text}")
        try:
            res = tavily.search(query=query_text, max_results=15, search_depth="advanced")
            for item in res.get("results", []):
                url = item.get("url")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    raw_results.append(item)
        except Exception as e:
            print(f" Search error on '{query_text}': {e}")

    print(f"Collected {len(raw_results)} unique job URLs.")
    return {"raw_results": raw_results}


# Node 3: Verification Agent
def verifier_node(state: AgentState) -> dict[str, Any]:
    print("\n[Verifier Tool] Checking URLs for expiration and active listings...")
    verified = []
    expired_count = 0
    closed_phrases = [
        "job closed",
        "no longer accepting applications",
        "position filled",
        "expired",
    ]

    candidates_url = state["raw_results"][:12]

    for item in candidates_url:
        url = item["url"]
        try:
            ext = tavily.extract(urls=[url])
            if not isinstance(ext, dict):
                continue

            results = ext.get("results", [])
            if not isinstance(results, list) or not results:
                print(f" [!] No content: {url}")
                continue

            first_result = results[0]
            if not isinstance(first_result, dict):
                continue

            raw_text = (results[0].get("raw_content") or "").lower()

            # Exclude expired or closed jobs
            if any(phrase in raw_text for phrase in closed_phrases):
                expired_count += 1
                continue

            item["page_content"] = raw_text[:2000]  # Pass context snippet to scorer
            verified.append(item)

            print(f" [✓] Active: {url}")

        except Exception as e:
            print(f" [!] Failed verification for {url} {type(e).__name__}: {e}")

    print(f"[Verifier] {len(verified)} active, {expired_count} expired/closed")
    return {"verified_jobs": verified}


# Node 4: Job Scoring & Report Agent
def scorer_node(state: AgentState) -> dict[str, Any]:
    print("\n[Scoring Agent] Evaluating candidate profile fit against verified listings...")

    jobs_summary = [
        {
            "title": job.get("title"),
            "url": job.get("url"),
            "snippet": job.get("content"),
            "extracted_text": job.get("page_content", "")[:1000],
        }
        for job in state["verified_jobs"]
    ]
    prompt = f"""

    Candidate Profile:
    {json.dumps(candidate_profile, indent=2)}

    Evaluate each verified job against this candidate.

    Evaluation criteria:
    - Target >= 70% skill overlap.
    - Reject if English is not the primary working language.
    - Reject if visa sponsorship is explicitly denied or restricted to local citizens.
    - Consider preferred job titles.
    - Consider primary technical skills.
    - Do not assume facts that are not present in the job listing.
    - If visa sponsorship is unclear, explicitly mark it as "Unknown".
    - Do not infer sponsorship merely because a role is remote.

    Verified Jobs :
    {json.dumps(jobs_summary, indent=2)}

    Output Requirement:
    - Assign fit_score as a percentage (0-100) based on skill overlap.
    - Be thorough and specific in the reason field.
    - Only include jobs that pass the evaluation criteria.
    """

    scorer_llm = llm.with_structured_output(JobReport)

    report = scorer_llm.invoke(
        [
            SystemMessage(
                content="You are an AI Career Agent evaluating job opportunities "
                "for a candidate profile."
            ),
            HumanMessage(content=prompt),
        ]
    )
    return {"final_report": render_markdown(report)}


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


def main():
    initial_state = {
        "queries": [],
        "raw_results": [],
        "verified_jobs": [],
        "final_report": "",
    }
    console = Console()

    output = graph.invoke(initial_state)
    print("FINAL JOB SEARCH REPORT")
    console.print(Markdown(output["final_report"]))


if __name__ == "__main__":
    main()
