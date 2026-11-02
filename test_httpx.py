import httpx
import os

os.environ["NO_PROXY"] = "localhost,127.0.0.1"

url = "http://127.0.0.1:7861"
print(f"Testing {url} with HEAD...")
try:
    r = httpx.head(url, timeout=3)
    print(f"Status: {r.status_code}")
except Exception as e:
    print(f"Error: {e}")

print(f"\nTesting {url} with GET...")
try:
    r = httpx.get(url, timeout=3)
    print(f"Status: {r.status_code}")
except Exception as e:
    print(f"Error: {e}")
