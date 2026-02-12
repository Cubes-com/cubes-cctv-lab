from mcp_server import mcp

try:
    # First, must call this
    app = mcp.streamable_http_app()
    
    # Now session_manager should be available
    sm = mcp.session_manager
    print(f"Session Manager: {sm}")
    for attr in dir(sm):
        if not attr.startswith("_"):
            print(f"- {attr}")
            
    # Check if app itself has lifespan
    # Starlette apps usually expose 'router' and handle lifespan via middleware stack
    print(f"\nApp Type: {type(app)}")
    print(f"App Attributes:")
    for attr in dir(app):
        if not attr.startswith("_"):
            print(f"- {attr}")
            
except Exception as e:
    print(f"Error: {e}")
