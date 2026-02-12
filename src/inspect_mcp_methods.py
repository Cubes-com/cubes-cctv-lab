from mcp_server import mcp

print("--- SSE App Routes ---")
sse_app = mcp.sse_app()
for route in sse_app.routes:
    print(f"Path: {route.path} | Name: {route.name} | Methods: {route.methods}")

print("\n--- Streamable HTTP App Routes ---")
http_app = mcp.streamable_http_app()
for route in http_app.routes:
    print(f"Path: {route.path} | Name: {route.name} | Methods: {route.methods}")
