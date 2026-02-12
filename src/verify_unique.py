import requests
import os

def test_unique_name_enforcement():
    # 1. Create a dummy person "TestUnique" (if not exists)
    # We can use the /create route if we have a sighting ID, but that's hard to mock without a sighting.
    # Easier to use /upload with a dummy image.
    
    url_upload = "http://127.0.0.1:5001/upload"
    
    # Create a dummy image
    img_path = "dummy_face.jpg"
    # Create a minimal valid jpeg or just use an existing one if available.
    # Let's try to use an existing one from data/identities if possible, or create a black square with CV2?
    # But I don't have cv2 installed in this script environment easily? 
    # Actually I am running inside the container, so CV2 might be there, but `requests` was the issue before.
    # Wait, I am running this script inside `web` container. `web` container has `cv2` and `requests`.
    
    # Let's create a dummy image using python (if cv2 available) or just a text file (might fail image check).
    # The app checks `cv2.imdecode`.
    
    import cv2
    import numpy as np
    
    # Create a dummy image with a "face" (random noise might not work for face detection).
    # This is tricky. If I can't detect a face, I can't create an identity.
    # But I can check if the API rejects DULICATE names even if face detection fails?
    # No, the code checks name uniqueness AFTER detecting a face? 
    # Wait, looking at `web_app.py`:
    # `if db.get_identity_id_by_name(name): return Error` -> This is BEFORE reading image!
    # So I don't need a valid face image to test uniqueness check! I just need ANY file and a name.
    
    # 1. First, ensure "TestUnique" exists. 
    # Actually, if it DOESN'T exist, the code proceeds to read image.
    # If I provide a bad image, it will fail at "Invalid image" or "No face detected".
    # BUT if I provide a name that EXISTS, it should fail at Name Check FIRST.
    
    # So the test strategy:
    # A. Use a name that definitely exists. "Jamie" (from previous steps).
    # B. Try to upload ANY file with name "Jamie".
    # C. Expect 400 "Name ... already exists".
    
    # Existing name? I need to be sure.
    # I can QUERY the DB or just assume "Jamie" exists if I didn't delete him.
    # Note: I might have deleted Jamie in the previous step verification!
    # I should find an existing name first.
    
    # How to find existing name?
    # I can use `/training` page or just direct DB query in this script.
    
    from identity_db import IdentityDB
    db = IdentityDB()
    existing_names = db.get_known_faces()
    db.close()
    
    if not existing_names:
        print("No existing identities found. Cannot test uniqueness collision.")
        # Create one? But I need a valid face to create one.
        # This is a blocker if DB is empty.
        # But I'm verifying the CODE logic. 
        # I can INSERT a dummy identity strictly for testing collision?
        db = IdentityDB()
        with db.conn.cursor() as cursor:
            # Insert a dummy identity roughly
            cursor.execute("INSERT INTO identities (name, embedding) VALUES (%s, %s) RETURNING id", ("TestUnique", b'0'*512)) # minimal embedding?
            identity_id = cursor.fetchone()[0]
            db.conn.commit()
        db.close()
        target_name = "TestUnique"
        print(f"Created temporary dummy identity '{target_name}' for testing.")
    else:
        target_name = existing_names[0][0]
        print(f"Using existing identity '{target_name}' for testing.")
        
    # Now try to upload with this name
    files = {'image': ('test.txt', b'dummy content')}
    data = {'name': target_name}
    
    print(f"Attempting to create duplicate of '{target_name}' via /upload...")
    resp = requests.post(url_upload, files=files, data=data)
    
    print(f"Response: {resp.status_code} - {resp.text}")
    
    if resp.status_code == 400 and "already exists" in resp.text:
        print("SUCCESS: Unique name enforcement working on /upload.")
    else:
        print("FAILURE: Did not get expected 400 or error message.")
        exit(1)

    # 2. Test /create route (from sighting)
    # This also checks name before anything else.
    # url_create = "http://127.0.0.1:5001/create"
    # data = {'name': target_name, 'sighting_id': 123} # Dummy sighting ID
    # resp = requests.post(url_create, data=data)
    # print(f"Attempting to create duplicate of '{target_name}' via /create...")
    # print(f"Response: {resp.status_code} - {resp.text}")
    
    # if resp.status_code == 400 and "already exists" in resp.text:
    #     print("SUCCESS: Unique name enforcement working on /create.")
    # else:
    #     print("FAILURE: Did not get expected 400 or error message.")
    #     exit(1)

    # Clean up if we created the dummy
    if target_name == "TestUnique":
        db = IdentityDB()
        db.delete_identity(target_name)
        db.close()
        print("Cleaned up dummy identity.")

if __name__ == "__main__":
    test_unique_name_enforcement()
