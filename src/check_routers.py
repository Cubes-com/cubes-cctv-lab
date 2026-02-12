from mcp_server import mcp
from fastapi import APIRouter

sse = mcp.sse_app()
http = mcp.streamable_http_app()

print(f"SSE Router: {getattr(sse, 'router', 'Not Found')}")
print(f"HTTP Router: {getattr(http, 'router', 'Not Found')}")

# Check if they are instances of something including router
try:
    if isinstance(sse.router, APIRouter) or hasattr(sse.router, 'routes'):
        print("SSE router looks valid.")
except:
    pass
