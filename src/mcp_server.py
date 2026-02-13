from mcp.server.fastmcp import FastMCP
from utils import get_db, map_camera_description, CAMERAS, get_camera_group, CAMERA_GROUPS
import json
import datetime
import zoneinfo

# Initialize FastMCP Server
mcp = FastMCP("cctv", host="0.0.0.0")

def format_response(content: str, structured_data: dict) -> str:
    """
    Standardized response format for all tools.
    Returns a JSON string containing both human-readable content and machine-readable data.
    """
    response = {
        "content": content,
        "structuredContent": structured_data
    }
    return json.dumps(response, default=str)

def get_utc_now():
    return datetime.datetime.now(datetime.timezone.utc)

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
    content = "Eglon House, 11a Sharpleshall Street, Cubes AI Ltd (Cubes)"
    return format_response(content, {"building": "Eglon House", "address": "11a Sharpleshall Street", "company": "Cubes AI Ltd"})

@mcp.tool()
def list_locations() -> str:
    """
    List all available location IDs and their associated cameras.
    Use this to understand what 'kitchen', 'lobby', etc. refer to.
    """
    # Combine CAMERA_GROUPS and standalone cameras
    locations = {}
    
    # Add groups first
    for group, cams in CAMERA_GROUPS.items():
        locations[group] = {
            "cameras": cams,
            "description": map_camera_description(group)
        }
        
    # Add standalone cameras if not covered (heuristic)
    for cam, desc in CAMERAS.items():
        if cam not in locations:
            # Check if it's already in a group? 
            # Simplified: just add it as a location
            locations[cam] = {
                "cameras": [cam],
                "description": desc
            }
            
    content = "Locations:\n" + "\n".join([f"- {k}: {v['description']}" for k,v in locations.items()])
    return format_response(content, {"locations": locations})

@mcp.tool()
def list_people() -> str:
    """
    List all known people and their last known location.
    Returns structured data with names and timestamps.
    """
    db = get_db()
    rows = db.get_all_identities()
    
    people = []
    lines = []
    
    for name, cam, ts in rows:
        desc = map_camera_description(cam) if cam else None
        people.append({
            "name": name,
            "last_seen_camera": cam,
            "last_seen_location": desc,
            "last_seen_ts": ts
        })
        
        time_str = str(ts) if ts else "Never seen"
        loc_str = desc if desc else "Unknown location"
        lines.append(f"- {name}: Last seen at {loc_str} ({time_str})")
        
    if not lines:
        return format_response("No people found.", {"people": []})
        
    return format_response("\n".join(lines), {"people": people})

@mcp.tool()
def list_cameras() -> str:
    """
    List all available cameras.
    """
    cameras_list = []
    lines = []
    for name, desc in CAMERAS.items():
        cameras_list.append({"id": name, "description": desc})
        lines.append(f"- {name}: {desc}")
        
    return format_response("\n".join(lines), {"cameras": cameras_list})

# --- Helper for Sighting Queries ---

def _execute_sighting_query(query_type: str, **kwargs):
    db = get_db()
    
    # Extract special args for metadata
    limit = kwargs.get('limit', 100)
    order = kwargs.get('order', 'DESC')
    location_id = kwargs.get('camera') # Mapped from location_id/camera
    
    # Expand location_id to camera list
    cameras_to_query = None
    if location_id:
        if isinstance(location_id, list):
             cameras_to_query = []
             for l in location_id:
                 cameras_to_query.extend(get_camera_group(l))
        else:
            cameras_to_query = get_camera_group(location_id)
            
    # Execute Query
    rows = db.query_sightings(
        name=kwargs.get('name'), 
        camera=cameras_to_query, 
        date_str=kwargs.get('date_str'), 
        limit=limit, 
        order=order,
        min_timestamp=kwargs.get('min_timestamp'),
        max_timestamp=kwargs.get('max_timestamp')
    )
    
    # Format Results
    sightings = []
    unique_people = set()
    first_ts = None
    last_ts = None
    
    for s_id, path, ts, cam, p_name, bbox in rows:
        desc = map_camera_description(cam)
        sightings.append({
            "ts": ts,
            "person": p_name,
            "camera_id": cam,
            "location_name": desc,
            "image_path": path
        })
        if p_name: unique_people.add(p_name)
        
        if not first_ts or ts < first_ts: first_ts = ts
        if not last_ts or ts > last_ts: last_ts = ts
        
    summary = {
        "count": len(sightings),
        "people": list(sorted(unique_people)),
        "first_ts": first_ts,
        "last_ts": last_ts
    }
    
    # Text Content Generation
    if not sightings:
        text = f"No results found for {query_type}."
    else:
        text = f"Found {len(sightings)} sightings.\n"
        text += f"People: {', '.join(summary['people'])}\n"
        # Add first few rows as examples
        text += "Recent:\n"
        for s in sightings[:5]:
             text += f"- [{s['ts']}] {s['person']} at {s['location_name']}\n"
             
    # Clean up kwargs for echo
    filters = kwargs.copy()
    if 'min_timestamp' in filters and filters['min_timestamp']: filters['min_timestamp'] = str(filters['min_timestamp'])
    if 'max_timestamp' in filters and filters['max_timestamp']: filters['max_timestamp'] = str(filters['max_timestamp'])

    return format_response(text, {
        "sightings": sightings,
        "summary": summary,
        "filters_applied": filters,
        "is_error": False
    })


@mcp.tool()
def query_sightings(
    query: str = None,
    name: str = None, 
    location_id: str = None, 
    date_str: str = None, 
    count: int = 20,
    order: str = "DESC"
) -> str:
    """
    General purpose sighting query.
    Use specific tools (first_seen, etc) if possible.
    location_id: Logical location (e.g. 'kitchen') or camera ID.
    """
    return _execute_sighting_query(
        "query_sightings",
        name=name,
        camera=location_id,
        date_str=date_str,
        limit=count,
        order=order
    )

@mcp.tool()
def first_seen(person: str, date_str: str = "today") -> str:
    """
    Find when a person was FIRST seen on a specific date.
    Returns the single earliest sighting.
    """
    return _execute_sighting_query(
        "first_seen",
        name=person,
        date_str=date_str,
        limit=1,
        order="ASC"
    )

@mcp.tool()
def last_seen(person: str, max_age_minutes: int = 60) -> str:
    """
    Find where a person was LAST seen within the last X minutes.
    Useful for "Where is X now?".
    """
    min_ts = datetime.datetime.now() - datetime.timedelta(minutes=max_age_minutes)
    return _execute_sighting_query(
        "last_seen",
        name=person,
        limit=1,
        order="DESC",
        min_timestamp=min_ts
    )

@mcp.tool()
def last_seen_on_date(person: str, date_str: str = "today") -> str:
    """
    Find when a person was LAST seen on a specific date.
    Useful for "When did X leave?".
    """
    return _execute_sighting_query(
        "last_seen_on_date",
        name=person,
        date_str=date_str,
        limit=1,
        order="DESC"
    )

@mcp.tool()
def where_has_been(person: str, date_str: str = "today") -> str:
    """
    Get a list of all locations a person has been to on a date.
    """
    return _execute_sighting_query(
        "where_has_been",
        name=person,
        date_str=date_str,
        limit=500, # Reasonable max
        order="ASC"
    )

@mcp.tool()
def people_in_location_now(location_id: str, max_age_minutes: int = 15) -> str:
    """
    List people seen in a location within the last X minutes.
    Useful for "Who is in the kitchen right now?".
    """
    min_ts = datetime.datetime.now() - datetime.timedelta(minutes=max_age_minutes)
    return _execute_sighting_query(
        "people_in_location_now",
        camera=location_id,
        min_timestamp=min_ts,
        limit=100,
        order="DESC"
    )

@mcp.tool()
def people_visited_location(location_id: str, date_str: str = "today") -> str:
    """
    List people who visited a location on a specific date.
    """
    return _execute_sighting_query(
        "people_visited_location",
        camera=location_id,
        date_str=date_str,
        limit=500,
        order="ASC"
    )

@mcp.tool()
def who_was_in_building(date_str: str = "today", start_time: str = None, end_time: str = None) -> str:
    """
    List everyone seen in the building on a date, optionally within a time range.
    start_time, end_time: "HH:MM" (24h format)
    """
    min_ts = None
    max_ts = None
    
    # Parse times if provided
    # Note: query_sightings (db) handles date_str logic for full day.
    # If partial day, we need to construct min/max_timestamp combining date and time.
    
    if start_time or end_time:
        # Determine base date
        base_date = datetime.date.today()
        if date_str == "yesterday":
            base_date = base_date - datetime.timedelta(days=1)
        elif date_str and date_str != "today":
            try:
                base_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                pass # Fallback to today or error?
        
        if start_time:
            try:
                t = datetime.datetime.strptime(start_time, "%H:%M").time()
                min_ts = datetime.datetime.combine(base_date, t)
            except: pass
            
        if end_time:
            try:
                t = datetime.datetime.strptime(end_time, "%H:%M").time()
                max_ts = datetime.datetime.combine(base_date, t)
            except: pass

    return _execute_sighting_query(
        "who_was_in_building",
        date_str=date_str,
        min_timestamp=min_ts,
        max_timestamp=max_ts,
        limit=1000,
        order="ASC"
    )


if __name__ == "__main__":
    print("Starting CCTV MCP Server (stdio)...")
    mcp.run()
