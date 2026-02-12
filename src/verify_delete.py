import requests

def test_delete_person():
    # 1. Create a dummy person first? 
    # Or just try to delete a non-existent one to see if it 400s or 200s (redirect)
    
    # Inside the container, we should use 127.0.0.1 to avoid localhost resolution issues
    url = "http://127.0.0.1:5001/delete_person"
    
    # Test missing name
    print(f"Testing missing name against {url}...")
    try:
        resp = requests.post(url, data={})
        print(f"Missing name response: {resp.status_code}")
        if resp.status_code != 400:
            print(f"Response text: {resp.text}")
        assert resp.status_code == 400
    except requests.exceptions.ConnectionError as e:
        print(f"Connection failed: {e}")
        return
    
    # Test deleting non-existent person (should redirect to index, code 302 -> 200)
    # Flask redirects are followed by requests by default
    resp = requests.post(url, data={"name": "NonExistentPerson123"})
    print(f"Delete non-existent response: {resp.status_code}")
    assert resp.status_code == 200
    assert "Dashboard" in resp.text or "Active Locations" in resp.text
    
    print("Verification script passed!")

if __name__ == "__main__":
    test_delete_person()
