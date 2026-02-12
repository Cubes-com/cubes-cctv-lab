import requests
import datetime
import json

BASE_URL = "http://api:8000"

def test_endpoint(url):
    try:
        resp = requests.get(f"{BASE_URL}{url}")
        print(f"GET {url} [{resp.status_code}]")
        if resp.status_code != 200:
            print(f"ERROR: {resp.text}")
            return None
        return resp.json()
    except Exception as e:
        print(f"FAILED {url}: {e}")
        return None

def verify_api():
    print("--- Verifying API Extensions ---")
    
    # 1. List People
    people = test_endpoint("/people")
    if people:
        print(f"Found {len(people)} people.")
        if len(people) > 0:
            print(f"Sample person: {people[0]}")
            
    # 2. List Cameras
    cameras = test_endpoint("/cameras")
    if cameras:
        print(f"Found {len(cameras)} cameras.")
        # Check for split cameras
        split_cam = next((c for c in cameras if "_left" in c['name'] or "_right" in c['name']), None)
        if split_cam:
            print(f"Confirmed split camera presence: {split_cam}")
        else:
            print("WARNING: No split cameras found in list (check cameras.yml configuration).")

    # 3. History
    if people and len(people) > 0:
        name = people[0]['name']
        history = test_endpoint(f"/people/{name}")
        if history:
            print(f"History for {name}: {len(history)} sightings.")
            
        # 4. Date Query
        today = "today"
        today_sightings = test_endpoint(f"/people/{name}/{today}")
        if today_sightings is not None:
            print(f"Sightings for {name} on {today}: {len(today_sightings)}")
            
        # 5. First/Last
        first = test_endpoint(f"/people/{name}/{today}/first")
        if first:
            print(f"First sighting today: {first[0]['timestamp']}")
        last = test_endpoint(f"/people/{name}/{today}/last")
        if last:
             print(f"Last sighting today: {last[0]['timestamp']}")

    if cameras and len(cameras) > 0:
        cam = cameras[0]['name']
        cam_hist = test_endpoint(f"/cameras/{cam}")
        if cam_hist:
             print(f"History for {cam}: {len(cam_hist)} sightings.")

if __name__ == "__main__":
    verify_api()
