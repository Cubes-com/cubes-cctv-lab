from mcp_server import mcp
from starlette.routing import Route, Mount

def print_remount_plan(app_name, app, prefix=""):
    print(f"--- Remounting {app_name} under {prefix} ---")
    for route in app.routes:
        if isinstance(route, Route):
            path = prefix + route.path
            print(f"ADD ROUTE: {path} -> {route.name} (Methods: {route.methods})")
        elif isinstance(route, Mount):
            path = prefix + route.path
            print(f"MOUNT: {path} -> {route.name} (App: {route.app})")
        else:
            print(f"UNKNOWN: {route}")

print_remount_plan("SSE App", mcp.sse_app(), "/mcp")
print_remount_plan("HTTP App", mcp.streamable_http_app(), "") 
