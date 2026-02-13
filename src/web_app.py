import cv2
import numpy as np
import insightface
from flask import Flask, render_template, request, redirect, send_file, jsonify, url_for
import os
import requests
from identity_db import IdentityDB
import datetime
import threading
import time

# Background Cleanup Task
def cleanup_loop():
    print("Starting cleanup loop...")
    while True:
        try:
            # Sleep first to allow startup
            time.sleep(60) 
            
            # Run cleanup
            db = IdentityDB()
            # Clean logs older than 24 hours
            deleted_rows, count = db.cleanup_old_logs(hours=24)
            if count > 0:
                print(f"CLEANUP: Deleted {count} old sighting logs.")
                # Optional: Delete files from disk?
                # for row in deleted_rows:
                #     path = row[0]
                #     if os.path.exists(path):
                #         os.remove(path)
            db.close()
            
            # Sleep for 1 hour
            time.sleep(3600)
        except Exception as e:
            print(f"Cleanup error: {e}")
            time.sleep(60)


# Initialize Face Analysis (Global)
# We lazily load it or load on startup. Let's load on startup for simplicity, 
# although it might slow down dev server reload.
print("Initializing Face Analysis...")
face_app = insightface.app.FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))
print("Face Analysis Initialized.")

app = Flask(__name__)
DB_PATH = "identities.db"
IDENTITIES_DIR = "data/identities"
os.makedirs(IDENTITIES_DIR, exist_ok=True)
API_HOST = os.environ.get("API_HOST", "http://api:8000")

def get_db():
    return IdentityDB()

@app.context_processor
def inject_counts():
    db = get_db()
    try:
        unknown_count = db.get_unknown_count()
    except Exception:
        unknown_count = 0
    db.close()
    return dict(unknown_count=unknown_count)

@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/locations/recent")
def recent_locations():
    # Proxy to API service, forwarding query parameters
    try:
        resp = requests.get(f"{API_HOST}/locations/recent", params=request.args)
        return jsonify(resp.json())
    except Exception as e:
        print(f"Error fetching from API: {e}")
        return jsonify([]), 500

@app.route("/training")
def training():
    db = get_db()
    
    # Get unassigned sightings
    sightings_raw = db.get_unassigned_sightings()
    sightings = []
    for s_id, path, ts, camera, bbox in sightings_raw:
        sightings.append({"id": s_id, "path": path, "timestamp": ts, "camera": camera, "bbox": bbox})
        
    # Get known identities for dropdown
    identities = [name for name, _ in db.get_known_faces()]
    identities.sort()
    
    db.close()
    return render_template("training.html", sightings=sightings, identities=identities)

@app.route("/person/<name>")
def person_view(name):
    db = get_db()
    
    identity_id = db.get_identity_id_by_name(name)
    if not identity_id:
        db.close()
        return f"Person {name} not found", 404
        
    # Get sightings (logs)
    sightings_raw = db.get_sightings_for_identity(identity_id, limit=100)
    sightings = []
    unconfirmed_count = 0
    for s_id, path, ts, is_perm, camera, bbox in sightings_raw:
        sightings.append({"id": s_id, "path": path, "timestamp": ts, "is_permanent": is_perm, "camera": camera, "bbox": bbox})
        if not is_perm:
            unconfirmed_count += 1
        
    # Get all identities for re-assignment dropdown
    identities = [n for n, _ in db.get_known_faces()]
    identities.sort()
    
    db.close()
    return render_template("person.html", name=name, sightings=sightings, identities=identities, unconfirmed_count=unconfirmed_count)


@app.route("/image/<int:sighting_id>")
def get_image(sighting_id):
    db = get_db()
    cursor = db.conn.cursor()
    cursor.execute("SELECT image_path FROM sightings WHERE id = %s", (sighting_id,))
    res = cursor.fetchone()
    db.close()
    
    if res:
        # Convert to absolute path to avoid Flask root_path issues
        image_path = os.path.abspath(res[0])
        if os.path.exists(image_path):
            return send_file(image_path)
    return "Not found", 404

@app.route("/create", methods=["POST"])
def create_identity():
    sighting_id = int(request.form.get("sighting_id"))
    name = request.form.get("name")
    
    db = get_db()
    db.create_identity_from_sighting(sighting_id, name)
    db.close()
    
    return redirect(request.form.get("next_url", "/training"))


@app.route("/assign", methods=["POST"])
def assign_identity():
    # Helper to clean parsed values
    sighting_id = request.form.get("sighting_id")
    if sighting_id: sighting_id = int(sighting_id)
    name = request.form.get("name")
    
    db = get_db()
    db.assign_sighting(sighting_id, name)
    db.close()
    
    # Check for AJAX
    if request.args.get("ajax") or request.form.get("ajax"):
        return jsonify({"status": "success", "message": f"Assigned to {name}"})
        
    return redirect(request.form.get("next_url", "/training"))

@app.route("/confirm_sighting", methods=["POST"])
def confirm_sighting():
    sighting_id = request.form.get("sighting_id")
    if sighting_id: sighting_id = int(sighting_id)
    
    db = get_db()
    # To confirm, we just set is_permanent=True (logic handled in DB or here)
    # We can use a custom DB method or just specific update
    with db.conn.cursor() as cursor:
        cursor.execute("UPDATE sightings SET is_permanent = TRUE WHERE id = %s", (sighting_id,))
    db.conn.commit()
    db.close()
    
    if request.args.get("ajax") or request.form.get("ajax"):
        return jsonify({"status": "success", "message": "Sighting confirmed"})
        
    # If not ajax, redirect back (referrer)
    return redirect(request.referrer or "/")

@app.route("/delete_sighting", methods=["POST"])
def delete_sighting():
    sighting_id = request.form.get("sighting_id")
    if sighting_id: sighting_id = int(sighting_id)
    next_url = request.form.get("next_url", "/")
    
    db = get_db()
    # Ensure we actually have a method for this or use raw SQL
    # check keys in db class first? Assuming delete_sighting doesn't exist in DB class yet?
    # I should check identity_db.py. 
    # But for now I'll use raw SQL if needed, or check if I added it.
    # checking identity_db.py ... I recall adding delete_identity, but not delete_sighting?
    # Let's assume raw SQL for now to be safe or minimal.
    with db.conn.cursor() as cursor:
        cursor.execute("DELETE FROM sightings WHERE id = %s", (sighting_id,))
    db.conn.commit()
    db.close()
    
    if request.args.get("ajax") or request.form.get("ajax"):
        return jsonify({"status": "success", "message": "Sighting deleted"})
        
    return redirect(next_url)

@app.route("/delete_person", methods=["POST"])
def delete_person():
    name = request.form.get("name")
    if not name:
        return "Missing name", 400
        
    db = get_db()
    
    # 1. Delete from DB and get file list
    files_to_delete = db.delete_identity(name)
    db.close()
    
    # 2. Delete files from disk
    count = 0
    for file_path in files_to_delete:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                count += 1
            except OSError as e:
                print(f"Error deleting file {file_path}: {e}")
                
    print(f"Deleted person {name} and {count} sighting files.")
    
    return redirect(url_for('dashboard'))


@app.route("/upload", methods=["POST"])
def upload_identity():
    name = request.form.get("name")
    file = request.files.get("image")
    
    if not name or not file:
        return "Missing name or file", 400
        
    db = get_db()
    if db.get_identity_id_by_name(name):
        db.close()
        return f"Error: Name '{name}' already exists. Please use a unique name.", 400
    
    # Read image
    in_memory_file = file.read()
    nparr = np.frombuffer(in_memory_file, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        db.close() # Should confirm db is closed if we return early? 
                   # Actually I opened db above. I should close it or move db open later.
                   # Let's close it before return.
        return "Invalid image", 400
        
    # Detect Face
    faces = face_app.get(img)
    if not faces:
        db.close()
        return "No face detected in photo", 400
        
    # Pick the largest face (or first)
    faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
    face = faces[0]

    # Save Reference Image
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.jpg"
    filepath = os.path.join(IDENTITIES_DIR, filename)
    cv2.imwrite(filepath, img)
    
    # Save Identity to DB (We already checked name existence, but technically race condition possible. Ignoring for now.)
    print(f"Creating new identity: {name}")
    db.save_identity(name, face.embedding)
        
    db.close()
    
    return redirect("/")

if __name__ == "__main__":
    # Start cleanup thread
    # Check if this is the reloader process (to avoid running twice)
    # WERKZEUG_RUN_MAIN is set by Werkzeug reloader in the child process
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        t = threading.Thread(target=cleanup_loop, daemon=True)
        t.start()
        
    app.run(host="0.0.0.0", port=5001, debug=True)

