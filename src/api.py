from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel
from typing import List, Optional, Any
import psycopg2
import datetime
import os
import yaml

from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

# Ensure src is in path or relative import works.
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import get_db, map_camera_description, CAMERAS
from mcp_server import mcp
from contextlib import asynccontextmanager

# Define lifespan to manage MCP session manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure http app is initialized so session_manager is available
    # (Checking if already initialized by route extraction below)
    # But safe to access.
    # Run the session manager context
    try:
        # We need to make sure streamable_http_app() was called at least once
        # which it is in the main body (see below).
        # We need to access mcp.session_manager.run()
        # Note: mcp object is imported.
        
        # Access session manager (it might be lazy, but we called app() below)
        # Verify if we need to call it effectively?
        # The route extraction logic below runs at module import time, so it's done.
        
        async with mcp.session_manager.run():
            yield
    except Exception as e:
        print(f"Error in MCP lifespan: {e}")
        yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    # User requested 404 for root to avoid confusion with actual dashboard on port 80
    raise HTTPException(status_code=404, detail="API Root. Dashboard is at port 80.")

@app.get("/mcp.json")
def get_mcp_config():
    if os.path.exists("mcp.json"):
        from fastapi.responses import FileResponse
        return FileResponse("mcp.json")
    return {"error": "mcp.json not found"}

# --- Models ---
class Sighting(BaseModel):
    name: str = "Unknown"
    camera: Optional[str] = None
    description: Optional[str] = None
    timestamp: str
    image_path: Optional[str] = None
    unconfirmed_count: int = 0

class PersonSummary(BaseModel):
    name: str
    last_seen_camera: Optional[str]
    last_seen_description: Optional[str]
    last_seen_timestamp: Optional[str]

class CameraSummary(BaseModel):
    name: str
    description: str

# --- Endpoints ---

@app.get("/locations/recent", response_model=List[Sighting])
def get_recent_locations(sort: str = "timestamp"):
    """
    Get list of people seen in the last 12 hours.
    sort: "timestamp" (default) or "name"
    """
    db = get_db()
    threshold = (datetime.datetime.now() - datetime.timedelta(hours=12))
    
    with db.conn.cursor() as cursor:
        # Get base locations
        cursor.execute("""
            SELECT pl.name, pl.camera, pl.timestamp, i.id
            FROM person_location pl
            LEFT JOIN identities i ON pl.name = i.name
            WHERE pl.timestamp > %s
            ORDER BY pl.timestamp DESC
        """, (threshold,))
        rows = cursor.fetchall()
        
    results = []
    for name, camera, timestamp, identity_id in rows:
        # Get unconfirmed count
        unconfirmed = 0
        if identity_id:
            unconfirmed = db.get_unconfirmed_count(identity_id)
            
        results.append({
            "name": name,
            "camera": camera,
            "description": map_camera_description(camera),
            "timestamp": str(timestamp),
            "unconfirmed_count": unconfirmed
        })

    # Sort results
    if sort == "name":
        results.sort(key=lambda x: x["name"].lower())
    else:
        # Default is timestamp DESC (already sorted by SQL, but let's be sure if we merged lists)
        results.sort(key=lambda x: x["timestamp"], reverse=True)

    return results

# Add unconfirmed_count to Sighting optionally
class SightingExtended(Sighting):
    unconfirmed_count: int = 0

@app.get("/people", response_model=List[PersonSummary])
def get_people():
    db = get_db()
    rows = db.get_all_identities()
    results = []
    for name, cam, ts in rows:
        results.append({
            "name": name,
            "last_seen_camera": cam,
            "last_seen_description": map_camera_description(cam) if cam else None,
            "last_seen_timestamp": str(ts) if ts else None
        })
    return results

@app.get("/cameras", response_model=List[CameraSummary])
def get_cameras():
    # Use the loaded descriptions
    results = []
    for name, desc in CAMERAS.items():
        results.append({"name": name, "description": desc})
    return results

# Generic Query Endpoint Handler
def handle_query(name=None, camera=None, date_str=None, order="DESC", limit=None):
    db = get_db()
    rows = db.query_sightings(name=name, camera=camera, date_str=date_str, order=order, limit=limit)
    
    results = []
    for s_id, path, ts, cam, p_name, bbox in rows:
        results.append({
            "name": p_name if p_name else "Unknown",
            "camera": cam,
            "description": map_camera_description(cam),
            "timestamp": str(ts),
            "image_path": path
        })
    return results

# /people/[person]
@app.get("/people/{name}", response_model=List[Sighting])
def get_person_history(name: str):
    return handle_query(name=name, limit=200)

# /cameras/[camera]
@app.get("/cameras/{camera}", response_model=List[Sighting])
def get_camera_history(camera: str):
    return handle_query(camera=camera, limit=200)

# /cameras/{camera}/people/{name}
@app.get("/cameras/{camera}/people/{name}", response_model=List[Sighting])
def get_camera_person_history(camera: str, name: str):
    return handle_query(camera=camera, name=name, limit=200)

# Date variants
@app.get("/people/{name}/{date_str}", response_model=List[Sighting])
def get_person_date(name: str, date_str: str):
    return handle_query(name=name, date_str=date_str)

@app.get("/cameras/{camera}/{date_str}", response_model=List[Sighting])
def get_camera_date(camera: str, date_str: str):
    return handle_query(camera=camera, date_str=date_str)

@app.get("/cameras/{camera}/people/{name}/{date_str}", response_model=List[Sighting])
def get_camera_person_date(camera: str, name: str, date_str: str):
    return handle_query(camera=camera, name=name, date_str=date_str)

# First/Last variants
@app.get("/people/{name}/{date_str}/first", response_model=List[Sighting])
def get_person_date_first(name: str, date_str: str):
    return handle_query(name=name, date_str=date_str, order="ASC", limit=1)

@app.get("/people/{name}/{date_str}/last", response_model=List[Sighting])
def get_person_date_last(name: str, date_str: str):
    return handle_query(name=name, date_str=date_str, order="DESC", limit=1)

@app.get("/cameras/{camera}/{date_str}/first", response_model=List[Sighting])
def get_camera_date_first(camera: str, date_str: str):
    return handle_query(camera=camera, date_str=date_str, order="ASC", limit=1)

@app.get("/cameras/{camera}/{date_str}/last", response_model=List[Sighting])
def get_camera_date_last(camera: str, date_str: str):
    return handle_query(camera=camera, date_str=date_str, order="DESC", limit=1)

@app.get("/cameras/{camera}/people/{name}/{date_str}/first", response_model=List[Sighting])
def get_camera_person_date_first(camera: str, name: str, date_str: str):
    return handle_query(camera=camera, name=name, date_str=date_str, order="ASC", limit=1)

@app.get("/cameras/{camera}/people/{name}/{date_str}/last", response_model=List[Sighting])
def get_camera_person_date_last(camera: str, name: str, date_str: str):
    return handle_query(camera=camera, name=name, date_str=date_str, order="DESC", limit=1)

# --- MCP Integration ---
try:
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from starlette.routing import Route, Mount
    
    # Trusted Host Config
    # Default to localhost to be safe, but allow user override
    allowed_hosts = os.getenv("MCP_HOST", "localhost").split(",")
    # Trim spaces
    allowed_hosts = [h.strip() for h in allowed_hosts]
    
    app.add_middleware(
        TrustedHostMiddleware, 
        allowed_hosts=allowed_hosts
    )

    # Remount MCP routes manually to avoid path stripping conflicts
    # between /mcp (Streamable) and /mcp/sse (SSE)
    
    # 1. SSE App (Legacy) -> Mount under /mcp prefix
    sse_app = mcp.sse_app()
    for route in sse_app.routes:
        if isinstance(route, Route):
            # e.g. /sse -> /mcp/sse
            path = f"/mcp{route.path}"
            methods = list(route.methods) if route.methods else None
            app.add_route(path, route.endpoint, methods=methods)
        elif isinstance(route, Mount):
            # e.g. /messages -> /mcp/messages
            path = f"/mcp{route.path}"
            app.mount(path, route.app)
            
    # 2. Streamable HTTP App -> Mount at root (it handles /mcp itself)
    http_app = mcp.streamable_http_app()
    for route in http_app.routes:
        if isinstance(route, Route):
            # e.g. /mcp -> /mcp
            # Streamable app usually handles GET/POST but methods might be None in route def
            methods = route.methods
            if not methods:
                methods = ["GET", "POST"]
            app.add_route(route.path, route.endpoint, methods=methods)
        elif isinstance(route, Mount):
            app.mount(route.path, route.app)
    
except ImportError:
    print("Warning: 'mcp' library not found. MCP integration disabled.")
except Exception as e:
    print(f"Warning: Failed to initialize MCP or Middleware: {e}")

if __name__ == "__main__":
    import uvicorn
    # Allow 0.0.0.0 binding inside container, validation handled by middleware
    uvicorn.run(app, host="0.0.0.0", port=8000)
