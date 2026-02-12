import yaml
import os
import sys

# Ensure we can import identity_db
# If running as script, sys.path might need adjustment, but usually we run from root.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from identity_db import IdentityDB
except ImportError:
    # Fallback if running from src directly maybe?
    pass

def get_db():
    return IdentityDB()

def load_camera_descriptions():
    try:
        # Assuming cameras.yml is in project root (.. from src)
        # But api.py assumed it was in current directory?
        # api.py is run from root in docker: CMD ["uvicorn", "src.api:app", ...]
        # So "cameras.yml" works.
        base_path = "cameras.yml"
        if not os.path.exists(base_path):
            # Try going up one level if we are inside src?
            base_path = "../cameras.yml"
            
        with open(base_path, "r") as f:
            config = yaml.safe_load(f)
            cameras = config.get("cameras", [])
            mapping = {}
            for cam in cameras:
                name = cam.get("name")
                desc = cam.get("description", name)
                mapping[name] = desc
                
                # Handle stereo split sub-cameras
                if cam.get("stereo_split"):
                    mapping[f"{name}_left"] = desc
                    mapping[f"{name}_right"] = desc
            return mapping
    except Exception as e:
        print(f"Error loading cameras.yml: {e}")
        return {}

CAMERAS = load_camera_descriptions()
CAMERA_GROUPS = {}

# Populate CAMERA_GROUPS by reloading or just iterating?
# We did it inside load_camera_descriptions but closed the scope.
# Let's just re-read or modify load_camera_descriptions to return both?
# Or just accept a global side effect (ugly but quick)?
# Refactoring load_camera_descriptions to populate the global.

def _refresh_camera_groups():
    global CAMERA_GROUPS
    CAMERA_GROUPS.clear()
    try:
        base_path = "cameras.yml"
        if not os.path.exists(base_path):
            base_path = "../cameras.yml"
            
        with open(base_path, "r") as f:
            config = yaml.safe_load(f)
            cameras = config.get("cameras", [])
            for cam in cameras:
                name = cam.get("name")
                if cam.get("stereo_split"):
                    # Maps "kitchen" -> ["kitchen", "kitchen_left", "kitchen_right"]
                    # We include the parent name just in case there are sightings with it.
                    CAMERA_GROUPS[name] = [name, f"{name}_left", f"{name}_right"]
    except Exception:
        pass

_refresh_camera_groups()

def get_camera_group(camera_name):
    """Returns a list of cameras if the name is a group, otherwise single item list."""
    if camera_name in CAMERA_GROUPS:
        return CAMERA_GROUPS[camera_name]
    return [camera_name]


def map_camera_description(camera_name):
    return CAMERAS.get(camera_name, camera_name)
