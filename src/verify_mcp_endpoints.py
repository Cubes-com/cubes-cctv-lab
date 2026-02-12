import requests
import sys

def check_url(url, expected_codes=[200]):
    try:
        print(f"Checking {url}...")
        # Use a short timeout
        resp = requests.get(url, timeout=2, allow_redirects=False)
        print(f"Status: {resp.status_code}")
        if resp.status_code in expected_codes:
            print("OK")
            return True
        else:
            print(f"FAIL: Expected {expected_codes}, got {resp.status_code}")
            print(f"Headers: {resp.headers}")
            print(f"Content: {resp.text[:500]}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

# Check Streamable (GET might be 405 Method Not Allowed if it only wants POST, or 200)
# Streamable usually supports GET? inspect methods said 'None' -> ['GET', 'POST']
print("\n--- Testing Streamable HTTP (/mcp) ---")
check_url("http://localhost:8000/mcp", expected_codes=[200, 405])

# Check SSE (/mcp/sse)
# Should be 200 (event stream)
print("\n--- Testing SSE (/mcp/sse) ---")
check_url("http://localhost:8000/mcp/sse", expected_codes=[200])

# Check Messages (/mcp/messages)
# Should be 405 (Method Not Allowed) for GET, since it's usually POST
print("\n--- Testing Messages (/mcp/messages) ---")
check_url("http://localhost:8000/mcp/messages", expected_codes=[405])
