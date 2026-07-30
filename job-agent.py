from typing import List
import yaml
import json
import requests
import os

from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
VLLM_KEY = os.getenv("OPENAI_API_KEY")
VLLM_HOST = os.getenv("VLLM_HOST")
VLLM_PORT = os.getenv("VLLM_PORT")

existing_no_proxy = os.environ.get("NO_PROXY", "")
os.environ["NO_PROXY"] = f"{VLLM_HOST},{existing_no_proxy}".strip(",")

from langchain.agents import create_agent
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from tavily import TavilyClient
import httpx

PROXY = os.getenv("HTTPS_PROXY")

session = requests.Session()
session.proxies = {"http": PROXY,"https": PROXY}
session.verify = "/etc/ssl/certs/ca-certificates.crt"

tavily = TavilyClient(
    api_key=os.environ["TAVILY_API_KEY"],
    session=session,
)

@tool
def search(query: str) -> str:
    """
    Searches the internet for active job postings and company details.
    Args:
        query: Specific search terms (e.g., 'site:greenhouse.io OR site:lever.co OR site:linkedin.com "SRE DevOps Engineer"Europe visa sponsorship').
    Return:
        JSON string of search exact URLs, titles, and snippets.
    """
    print(f"searching for {query}")
    results = tavily.search(
        query=query,
        max_results=15,
        search_depth="advanced",
    )
    return json.dumps(results)

@tool
def verify_job_url(url: str) -> str:
    """
    open a job URL and extracts its raw text to check if the position is active and offers visa sponsorship.
    """
    try:
        response = tavily.extract(urls=[url])
        return json.dumps(response)
    except Exception as e:
        return f"Could not verify URL: {str(e)}"

llm = ChatOpenAI(
    model="Google/Gemma-4-31B-it",
    base_url=f"http://{VLLM_HOST}:{VLLM_PORT}/v1",
    api_key=VLLM_KEY,
)
with open("candidate_profile.yaml", "r") as f:
    CANDIDATE_PROFILE = yaml.safe_load(f)

system_prompt = (
    "You are an expert AI Job Hunter. \n"
    "STRICT RULES:\n"
    "1. DIRECT LINKS ONLY: You MUST use the exact 'url' field returned by the tool. Never invent, reuse, or hallucinate URLs.\n"
    "2. VERIFY EXPIRATION: Use the `verify_job_url` tool on potential job links. If the page contains text like 'Job Closed', 'No longer accepting applications', or 'Expired', DISCARD the role.\n"
    "3. HIGH INTENT SEARCH: Search job portals directly using terms like 'site:lever.co', 'site:greenhouse.io', or 'site:workable.com'."
    )

tools = [search, verify_job_url]
agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)

def main():
    print("Job agent start searching ...")
    candidate_profile = """
    Skills: Kubernetes, AWS, CI/CD, Python, Prometheus, Docker.
    Experience: 10 years in SRE / Infrastructure Enginerring.
    Language: Professinal English, Starter German, native Farsi.
    """

    query = f"""
    Search for open SRE and DevOps position in Europe posted in the last 7 days.

    Execution Workflow:
    1. Search for posting on major ATS platforms (e.g. site:greenhouse.io, site:lever.co, site:workable.com).
    2. Run `verify_job_url` on top results to confirm they are still OPEN and explicitly offer English-speaking environments with visa sponsorship.
    3. return 7 fully verified, active job postings.

    Candidate Profile:
    {candidate_profile}

    Format output with a markdown table containing: Job Title, Company, Location, Visa Status, and Direct Link.
    """
    result = agent.invoke({
        "messages": [HumanMessage(content=query)]
    })
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
