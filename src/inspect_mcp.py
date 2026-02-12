from mcp.server.fastmcp import FastMCP
import inspect

mcp = FastMCP("cctv")
print("Attributes of mcp:")
for attr in dir(mcp):
    if not attr.startswith("_"):
        print(f"- {attr}")

print("\nAttributes of mcp.sse_app():")
# sse_app() returns a Starlette/FastAPI app
app = mcp.sse_app()
for attr in dir(app):
    if not attr.startswith("_"):
        print(f"- {attr}")
