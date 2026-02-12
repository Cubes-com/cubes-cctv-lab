from mcp_server import mcp
import inspect

try:
    sm = mcp.session_manager
    print(f"Session Manager: {sm}")
    for attr in dir(sm):
        if not attr.startswith("_"):
            print(f"- {attr}")
            
    # Check if there is a 'lifespan' on StreamableHTTPASGIApp
    app = mcp.streamable_http_app()
    print(f"\nStreamable App: {app}")
    print(f"Has lifespan? {hasattr(app, 'lifespan')}")
    
except Exception as e:
    print(f"Error: {e}")
