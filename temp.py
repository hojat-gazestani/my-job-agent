import json
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from tavily import TavilyClient

load_dotenv()
VLLM_KEY = os.getenv("OPENAI_API_KEY")
VLLM_HOST = os.getenv("VLLM_HOST")
VLLM_PORT = os.getenv("VLLM_PORT")
VLLM_MODEL = os.getenv("VLLM_MODEL")

# Initialize Tavily
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


@tool
def search(query: str) -> str:
    """
    Searches the internet for active job postings and company details.
    Args:
        query: Concise keyword search terms (e.g., 'DevOps SRE English visa sponsorship Europe')
    Returns:
        JSON string of search results.
    """
    print(f"Searching for: {query}")
    results = tavily.search(query=query, max_results=5, search_depth="advanced")
    return json.dumps(results)


# Initialize vLLM / OpenAI client
llm = ChatOpenAI(
    model="Google/Gemma-4-31B-it",
    base_url=f"http://{VLLM_HOST}:{VLLM_PORT}/v1",
    api_key=VLLM_KEY,
)

system_prompt = (
    "You are an expert AI Career Assistant. "
    "When searching for jobs, convert complex user criteria into short, "
    "high-intent keyword queries. "
    "Filter out expired postings and match job criteria against "
    "the provided candidate profile."
)

tools = [search]
agent = create_react_agent(model=llm, tools=tools, state_modifier=system_prompt)


def main():
    print("Job agent starting search...")

    # Pass concrete skills so the LLM can score fit
    candidate_profile = """
    Skills: Kubernetes, Terraform, AWS/GCP, CI/CD, Python, Prometheus, Docker.
    Experience: 4 years in SRE / Infrastructure Engineering.
    Language: Professional English.
    """

    query = f"""
    Search for open SRE and DevOps positions in Europe posted in the last 7 days.

    Constraints:
    - English-speaking role with visa sponsorship available.
    - Position must still be actively accepting applications.
    - Match against candidate profile:
    {candidate_profile}

    Format output with: Job Title, Company, Location, Visa Status, and Direct Link.
    """

    result = agent.invoke({"messages": [HumanMessage(content=query)]})

    # Print final agent answer
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
