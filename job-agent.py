import json
import os
from typing import Any, Dict, List, TypedDict

import requests
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()
VLLM_KEY = os.getenv("OPENAI_API_KEY")
VLLM_HOST = os.getenv("VLLM_HOST")
VLLM_PORT = os.getenv("VLLM_PORT")

existing_no_proxy = os.environ.get("NO_PROXY", "")
os.environ["NO_PROXY"] = f"{VLLM_HOST},{existing_no_proxy}".strip(",")

import httpx
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END
from tavily import TavilyClient

PROXY = os.getenv("HTTPS_PROXY")

session = requests.Session()
session.proxies = {"http": PROXY, "https": PROXY}
session.verify = "/etc/ssl/certs/ca-certificates.crt"

tavily = TavilyClient(
    api_key=os.environ["TAVILY_API_KEY"],
    session=session,
)

llm = ChatOpenAI(
    model="Google/Gemma-4-31B-it",
    base_url=f"http://{VLLM_HOST}:{VLLM_PORT}/v1",
    api_key=VLLM_KEY,
)


# Load Candidate profile
with open("candidate_profile.yaml", "r") as f:
    CANDIDATE_PROFILE = yaml.safe_load(f)


class AgentState(TypedDict):
    """
    Lnaggraph shared state
    """

    queries: List[str]
    raw_results: List[Dict[str, Any]]
    verified_jobs: List[Dict[str, Any]]
    final_report: str


class GeneratedQueries(BaseModel):
    """
    Pydantic Schema for Planner
    """

    queries: List[str] = Field(
        description="List of targeted queries derived from candidated profile."
    )


# Node 1: PLanner Agent
def planner_node(state: AgentState) -> Dict[str, Any]:
    print("\n[Planner Agent] Generating search queries from candidate profile ...")

    planner_llm = llm.with_structured_output(GeneratedQueries)

    prompt = f"""
    You are an expert Executive Tech Recruiter.
    Analyze this candidate profile:
    {json.dumps(CANDIDATE_PROFILE, indent=2)}

    Generate 6 targeted search queries to find current job listings in Europe with visa sponsorship.
    Derive specific keyword combinations (e.g., using ATS domains like site:greenhouse.io or site:lever.co combined with specific stack variants).

    Rules:
    - Target diverse roles that match their experience (e.g., Platform, SRE, Developer Productivity, Infrastructure).
    """

    res = planner_llm.invoke([SystemMessage(content=prompt)])
    print(f"Generated {len(res.queries)} targeted queries.")
    return {"queries": res.queries}


# Node 2: Search Tool Executor
def search_node(state: AgentState) -> Dict[str, Any]:
    print("\n[Search Tool] Executing Tavily searchs...")
    raw_results = []
    seen_urls = set()

    for query_text in state["queries"]:
        print(f" Searching : {query_text}")
        try:
            res = tavily.search(
                query=query_text, max_results=15, search_depth="advanced"
            )
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
def verifier_node(state: AgentState) -> Dict[str, Any]:
    print("\n[Verfier Tool] Checking URLs for expiration and active listings...")
    verified = []

    candidates_url = state["raw_results"][:12]

    for item in candidates_url:
        url = item["url"]
        try:
            ext = tavily.extract(urls=[url])
            results = ext.get("results", [])
            if results:
                raw_text = results[0].get("raw_content", "").lower()
                # Exclude expired or closed jobs
                if any(
                    phrase in raw_text
                    for phrase in [
                        "job closed",
                        "no longer accepting applications",
                        "position filled",
                        "expired",
                    ]
                ):
                    print(f" [X] Expired/Closed: {url}")
                    continue

                item["page_content"] = raw_text[:2000]  # Pass context snippet to scorer
                verified.append(item)
                print(f" [✓] Active: {url}")
        except Exception as e:
            print(f" [!] Failed verification for {url}: {e}")

    return {"verified_jobs": verified}


# Node 4: Job Scoring & Report Agent
def scorer_node(state: AgentState) -> Dict[str, Any]:
    print(
        "\n[Scoring Agent] Evaluating condidate profile fit against verfied listings..."
    )

    jobs_summary = []
    for j in state["verified_jobs"]:
        jobs_summary.append(
            {
                "title": j.get("title"),
                "url": j.get("url"),
                "snippet": j.get("content"),
                "extracted_text": j.get("page_content", "")[:1000],
            }
        )

    prompt = f"""
    You are an AI Career Agent evaluating job opportunities for a candidate.

    Candidate Profile:
    Score each of the verified jobs below based on profile overlap.

    Evaluation Criteria:
    - Target >= 70% skill overlap.
    - Reject if English is not the primary language.
    - Reject if visa sponsorship is explicitly denied or restricted to local citizens.
    - Factor in match quality against preferred job titles and primary skills.

    Verified Jobs to Score:
    {json.dumps(jobs_summary, indent=2)}

    Output Requirement:
    Return a cleanly formatted Markdown report containing:
    1. A summary table of top-matching roles (Job Title, Company, Location, Fit Score %, Visa Status, Direct Link).
    2. A brief breakdwon explaining why each role was chosen or rejected based on profile match.
    """

    response = llm.invoke([SystemMessage(content=prompt)])
    return {"final_report": response.content}


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

    output = graph.invoke(initial_state)
    print("\n" + "=" * 50)
    print("FINAL JOB SEARCH REPORT")
    print("=" * 50 + "\n")
    print(output["final_report"])


if __name__ == "__main__":
    main()
