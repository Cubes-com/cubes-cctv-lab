from mcp_server import mcp
from starlette.middleware.trustedhost import TrustedHostMiddleware

http_app = mcp.streamable_http_app()

print("Middleware:")
for m in http_app.user_middleware:
    print(f"- {m.cls}")
    if m.cls == TrustedHostMiddleware:
        print(f"  Options: {m.options}")

print("\nRoutes:")
from starlette.routing import Route, Mount
for r in http_app.routes:
    if isinstance(r, Route):
        print(f"Route: {r.path} -> {r.endpoint}")
    elif isinstance(r, Mount):
        print(f"Mount: {r.path} -> {r.app}")
