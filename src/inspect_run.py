from mcp_server import mcp
import inspect

try:
    mcp.streamable_http_app()
    sm = mcp.session_manager
    
    print(f"Run signature: {inspect.signature(sm.run)}")
    
    print(f"Has __aenter__? {hasattr(sm, '__aenter__')}")
    print(f"Has __enter__? {hasattr(sm, '__enter__')}")
    
except Exception as e:
    print(f"Error: {e}")
