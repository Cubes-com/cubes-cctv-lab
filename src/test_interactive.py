import requests
import json
import random

BASE_URL = "http://127.0.0.1:5001"

def test_interactive_features():
    # 1. We need a sighting ID to test.
    # We can try to list sightings from /training page or query DB if we had a direct DB client here.
    # But inside the container, we can use identity_db to check for a sighting.
    
    from identity_db import IdentityDB
    db = IdentityDB()
    
    # Ensure we have at least one sighting
    # If not, create a dummy one.
    with db.conn.cursor() as cursor:
        cursor.execute("INSERT INTO sightings (image_path, embedding, timestamp) VALUES (%s, %s, NOW()) RETURNING id", ("dummy.jpg", b'0'*512))
        sighting_id = cursor.fetchone()[0]
        db.conn.commit()
    
    print(f"Created dummy sighting ID: {sighting_id}")
    
    # 2. Test AJAX Confirm
    print("Testing AJAX Confirm...")
    resp = requests.post(f"{BASE_URL}/confirm_sighting", data={'sighting_id': sighting_id, 'ajax': '1'})
    print(f"Confirm Response: {resp.status_code} {resp.text}")
    assert resp.status_code == 200
    assert resp.json()['status'] == 'success'
    
    # Verify in DB
    with db.conn.cursor() as cursor:
        cursor.execute("SELECT is_permanent FROM sightings WHERE id = %s", (sighting_id,))
        is_perm = cursor.fetchone()[0]
        assert is_perm is True
        print("Verified Confirmation in DB.")

    # 3. Test AJAX Assign (to a name)
    # Create the identity first
    print("Creating dummy identity 'TestAssign'...")
    with db.conn.cursor() as cursor:
        cursor.execute("INSERT INTO identities (name, embedding) VALUES (%s, %s) RETURNING id", ("TestAssign", b'0'*512))
        test_assign_id = cursor.fetchone()[0]
        db.conn.commit()
    
    print("Testing AJAX Assign...")
    resp = requests.post(f"{BASE_URL}/assign", data={'sighting_id': sighting_id, 'name': 'TestAssign', 'ajax': '1'})
    print(f"Assign Response: {resp.status_code} {resp.text}")
    assert resp.status_code == 200
    assert resp.json()['status'] == 'success'
    
    # Verify in DB
    with db.conn.cursor() as cursor:
        cursor.execute("SELECT identity_id FROM sightings WHERE id = %s", (sighting_id,))
        ident_id = cursor.fetchone()[0]
        assert ident_id == test_assign_id
        
        cursor.execute("SELECT name FROM identities WHERE id = %s", (ident_id,))
        name = cursor.fetchone()[0]
        assert name == 'TestAssign'
        print("Verified Assignment in DB.")

    # 4. Test AJAX Delete
    print("Testing AJAX Delete...")
    resp = requests.post(f"{BASE_URL}/delete_sighting", data={'sighting_id': sighting_id, 'ajax': '1'})
    print(f"Delete Response: {resp.status_code} {resp.text}")
    assert resp.status_code == 200
    assert resp.json()['status'] == 'success'
    
    # Verify in DB
    with db.conn.cursor() as cursor:
        cursor.execute("SELECT id FROM sightings WHERE id = %s", (sighting_id,))
        res = cursor.fetchone()
        assert res is None
        print("Verified Deletion in DB.")
        
    # Cleanup Identity "TestAssign"
    try:
        db.delete_identity("TestAssign")
    except:
        pass
    db.close()
    print("Cleanup done.")

if __name__ == "__main__":
    test_interactive_features()
