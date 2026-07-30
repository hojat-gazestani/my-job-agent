import os
from dotenv import load_dotenv
import requests
from tavily import TavilyClient

load_dotenv()

PROXY = os.getenv("HTTPS_PROXY")

session = requests.Session()
session.proxies = {
    "http": PROXY,
    "https": PROXY,
}

# Optional: if your proxy uses a custom CA
session.verify = "/etc/ssl/certs/ca-certificates.crt"

tavily = TavilyClient(
    api_key=os.environ["TAVILY_API_KEY"],
    session=session,
)

print("Searching...")

try:
    result = tavily.search(
        query="What is the weather in Tokyo?",
        max_results=2,
    )

    print("Success!")
    print(result)

except Exception as e:
    print(f"Failed: {e}")
