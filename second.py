from typing import List
import requests
import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

# Set NO_PROXY FIRST before any LLM client is initialized
VLLM_HOST = "192.168.12.129"
existing_no_proxy = os.environ.get("NO_PROXY", "")
os.environ["NO_PROXY"] = f"{VLLM_HOST},{existing_no_proxy}".strip(",")

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from tavily import TavilyClient
import httpx

PROXY = os.getenv("HTTPS_PROXY")

# Tavily: routes through proxy
session = requests.Session()
session.proxies = {"http": PROXY, "https": PROXY}
session.verify = "/etc/ssl/certs/ca-certificates.crt"
tavily = TavilyClient(
    api_key=os.environ["TAVILY_API_KEY"],
    session=session,
)

@tool
def search(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"searching for {query}")
    return tavily.search(query=query)

# LLM: NO_PROXY ensures this bypasses the proxy
llm = ChatOpenAI(
    model="Google/Gemma-4-31B-it",
)

tools = [search]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({
        "messages": [HumanMessage(content="What is the weather in Tokyo?")]
    })
    print(result)

if __name__ == "__main__":
    main()
