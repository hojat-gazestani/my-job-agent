import os

import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

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
