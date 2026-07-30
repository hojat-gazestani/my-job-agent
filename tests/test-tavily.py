import os

import requests
from dotenv import load_dotenv

load_dotenv()


def test_proxy():
    proxies = {}
    if os.getenv("HTTP_PROXY"):
        proxies["http"] = os.getenv("HTTP_PROXY")
    if os.getenv("HTTPS_PROXY"):
        proxies["https"] = os.getenv("HTTPS_PROXY")

    print(f"Using proxies: {proxies}")

    # Test with a known working endpoint (Google's DNS or similar)
    test_urls = [
        "https://www.google.com",
        "https://api.tavily.com",
        "https://httpbin.org/get",  # This always works for testing
    ]

    for url in test_urls:
        try:
            print(f"\nTesting URL: {url}")
            response = requests.get(
                url, proxies=proxies if proxies else None, timeout=10, verify=False
            )
            print(f"✅ Success! Status: {response.status_code}")
            print(f"Response preview: {response.text[:200]}")
            return True
        except Exception as e:
            print(f"❌ Failed: {e}")

    return False


# Test with Tavily API using POST
def test_tavily_api():
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("❌ TAVILY_API_KEY not found in .env")
        return False

    proxies = {}
    if os.getenv("HTTP_PROXY"):
        proxies["http"] = os.getenv("HTTP_PROXY")
    if os.getenv("HTTPS_PROXY"):
        proxies["https"] = os.getenv("HTTPS_PROXY")

    url = "https://api.tavily.com/search"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"query": "weather in Tokyo", "search_depth": "basic", "max_results": 1}

    try:
        print("\nTesting Tavily API...")
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            proxies=proxies if proxies else None,
            timeout=30,
            verify=False,
        )
        print(f"Tavily API Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Tavily API test successful!")
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"❌ Tavily API failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Tavily API test failed: {e}")
        return False


if __name__ == "__main__":
    print("=== Testing Proxy Configuration ===")
    proxy_works = test_proxy()
    print("\n=== Testing Tavily API ===")
    tavily_works = test_tavily_api()

    if proxy_works and tavily_works:
        print("\n✅ All tests passed! Your configuration is working.")
    else:
        print("\n❌ Some tests failed. Check your configuration.")
