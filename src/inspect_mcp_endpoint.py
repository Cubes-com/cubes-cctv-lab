from mcp_server import mcp
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.routing import Route

http_app = mcp.streamable_http_app()

print("Checking Routes for Endpoint Middleware:")
for r in http_app.routes:
    if isinstance(r, Route) and r.path == "/mcp":
        endpoint = r.endpoint
        print(f"Endpoint: {endpoint}")
        print(f"Type: {type(endpoint)}")
        
        # Check for middleware attributes
        if hasattr(endpoint, "user_middleware"):
            print("Endpoint has user_middleware:")
            for m in endpoint.user_middleware:
                print(f"- {m.cls}")
                if m.cls == TrustedHostMiddleware:
                    print(f"  Options: {m.options}")
                    
        if hasattr(endpoint, "middleware_stack"):
             print("Endpoint has middleware_stack")
