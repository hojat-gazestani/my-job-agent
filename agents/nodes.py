import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from cli.display import render_markdown
from config.clients import llm, tavily
from config.profile import get_profile
from models.models import JobReport

from .state import AgentState, GeneratedQueries

candidate_profile = get_profile()


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
