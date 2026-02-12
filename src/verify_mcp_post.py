import requests
import sys

def check_url(url, method="GET", expected_codes=[200], headers=None, data=None):
    try:
        print(f"Checking {method} {url}...")
        if method == "GET":
            resp = requests.get(url, timeout=2, allow_redirects=False, headers=headers)
        elif method == "POST":
            resp = requests.post(url, timeout=2, allow_redirects=False, headers=headers, json=data)
            
        print(f"Status: {resp.status_code}")
        if resp.status_code in expected_codes:
            print("OK")
            # print(f"Headers: {resp.headers}")
            # print(f"Content: {resp.text[:500]}")
            return True
        else:
            print(f"FAIL: Expected {expected_codes}, got {resp.status_code}")
            print(f"Headers: {resp.headers}")
            print(f"Content: {resp.text[:500]}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

# 1. Test Streamable HTTP POST with correct headers (Initialize)
print("\n--- Testing Streamable HTTP POST (/mcp) with Correct Headers ---")
headers = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json"
}
data = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05", # Using a recent version
        "clientInfo": {"name": "test-client", "version": "1.0"},
        "capabilities": {}
    }
}
check_url("http://localhost:8000/mcp", method="POST", expected_codes=[200, 202], headers=headers, data=data)

# 2. Test without headers (Reproduction of user error)
print("\n--- Testing Streamable HTTP POST (/mcp) without Accept Header ---")
check_url("http://localhost:8000/mcp", method="POST", expected_codes=[406], headers={"Content-Type": "application/json"}, data=data)
