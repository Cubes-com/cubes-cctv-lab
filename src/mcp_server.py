from mcp.server.fastmcp import FastMCP
from utils import get_db, map_camera_description, CAMERAS, get_camera_group

# Initialize FastMCP Server
# Set host="0.0.0.0" to disable FastMCP's automatic localhost DNS rebinding protection,
# which otherwise enforces strict host checking (TrustedHostMiddleware).
mcp = FastMCP("cctv", host="0.0.0.0")

@mcp.resource("cctv://info")
def get_info() -> str:
    """Get information about the building and company."""
    return "Eglon House, 11a Sharpleshall Street, Cubes AI Ltd (Cubes)"

@mcp.tool()
def get_building_info() -> str:
    """
    Get metadata about the building and company.
    This is useful for context (e.g. "Where is this?").
    """
    return "Eglon House, 11a Sharpleshall Street, Cubes AI Ltd (Cubes)"

@mcp.tool()
def list_people() -> str:
    """
    List all known people and their last known location.
    Returns a formatted string summary.
    """
    db = get_db()
    rows = db.get_all_identities()
    if not rows:
        return "No people found."
    
    lines = []
    for name, cam, ts in rows:
        desc = map_camera_description(cam) if cam else "Unknown location"
        time_str = str(ts) if ts else "Never seen"
        lines.append(f"- {name}: Last seen at {desc} ({time_str})")
    return "\n".join(lines)

@mcp.tool()
def list_cameras() -> str:
    """
    List all available cameras.
    Returns a formatted string summary.
    """
    lines = []
    for name, desc in CAMERAS.items():
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)

@mcp.tool()
def query_sightings(
    query: str,
    name: str = None, 
    camera: str = None, 
    date_str: str = None, 
    count: int = 20,
    order: str = "DESC"
) -> str:
    """
    Flexible tool to query sightings. Use this for questions like:
    'Where was Katy last seen?', 'When did Jake leave?', 'Who is in the kitchen?'
    
    Args:
        query: The original user question (optional context).
        name: The person's name (e.g. 'Katy').
        camera: The camera/location name (e.g. 'kitchen').
        date_str: 'today', 'yesterday', or YYYY-MM-DD.
        count: Number of results to return (default 20).
        order: Sort order, 'ASC' or 'DESC' (default 'DESC').
    """
    db = get_db()
    
    # Sanitize order
    if order.upper() not in ["ASC", "DESC"]:
        order = "DESC"
        
    # Expand camera group (e.g. 'kitchen' -> ['kitchen', 'kitchen_left', 'kitchen_right'])
    cameras_to_query = camera
    if camera:
        cameras_to_query = get_camera_group(camera)
        
    rows = db.query_sightings(name=name, camera=cameras_to_query, date_str=date_str, limit=count, order=order)
    
    if not rows:
        return "No sightings found matching criteria."
        
    results = []
    for s_id, path, ts, cam, p_name, bbox in rows:
        desc = map_camera_description(cam)
        results.append(f"[{ts}] {p_name} at {desc} ({cam})")
        
    return "\n".join(results)

if __name__ == "__main__":
    # fastmcp.run() defaults to stdio if no args provided?
    # Or explicitly: mcp.run(transport='stdio')
    print("Starting CCTV MCP Server (stdio)...")
    mcp.run()
