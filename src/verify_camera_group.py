import requests
import json
import sys

def check_query(camera_name):
    url = "http://localhost:8000/mcp"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"
    }
    
    # 1. Initialize
    print("Initializing session...")
    init_payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05", # Using a recent version
            "clientInfo": {"name": "test-client", "version": "1.0"},
            "capabilities": {}
        }
    }
    
    try:
        resp = requests.post(url, headers=headers, json=init_payload)
        session_id = resp.headers.get("mcp-session-id")
        if not session_id:
            print(f"Failed to get session ID. Status: {resp.status_code}")
            print(resp.text)
            return

        print(f"Session ID: {session_id}")
        headers["mcp-session-id"] = session_id
        
        # 2. Call Tool
        payload = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "query_sightings",
                "arguments": {
                    "query": f"Who is in the {camera_name}?",
                    "camera": camera_name,
                    "count": 5
                }
            }
        }
    
        print(f"Querying camera: {camera_name}...")
        resp = requests.post(url, headers=headers, json=payload, stream=True)
        print(f"Status: {resp.status_code}")
        
        content = ""
        for line in resp.iter_lines():
            if line:
                decoded = line.decode('utf-8')
                # Extract data from SSE event if needed, or raw line
                if decoded.startswith("data: "):
                     content += decoded[6:] # Strip 'data: ' prefix
                else:
                     content += decoded + "\n"
                
        print("Response Snippet:")
        print(content[:1000])
        
        # Check for our data
        if "kitchen_left" in content or "kitchen_right" in content:
            print("SUCCESS: Found split camera results in parent query.")
        else:
             print("FAILURE/WARNING: No split camera names in output.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_query("kitchen")
