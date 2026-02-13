from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from pydantic import BaseModel
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis
import os
import sys
import datetime
from typing import List, Optional

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from identity_db import IdentityDB

app = FastAPI(title="CCTV Face Service", description="API for edge device face detection and recognition")

# --- Globals ---
face_app = None
db = None

# --- Models ---
class RecognizeRequest(BaseModel):
    embedding: List[float]
    camera: str
    threshold: float = 0.5  # Default threshold

class DetectResponse(BaseModel):
    status: str
    name: str
    sighting_id: int
    confidence: Optional[float] = None
    embedding: Optional[List[float]] = None
    message: Optional[str] = None

# --- Initialization ---
@app.on_event("startup")
async def startup_event():
    global face_app, db
    
    # Initialize DB
    print("Connecting to DB...")
    db = IdentityDB()
    
    # Initialize InsightFace
    print("Initializing InsightFace...")
    # Consistent with analyse_rtsp.py
    face_app = FaceAnalysis(name='buffalo_s', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    print("Service Ready.")

# --- Helpers ---
def process_sighting(embedding, camera_name):
    """
    Common logic for handling a face embedding:
    1. Identify person.
    2. Check for active sighting.
    3. Merge or Create.
    """
    # 1. Identify
    # get_best_match returns (name, similarity)
    name, similarity = db.get_best_match_with_score(embedding)
    
    # Threshold check? 
    # get_best_match already handles threshold logic if we want, 
    # but currently it just returns best match.
    # We should probably define a threshold here or in DB.
    # analyse_rtsp uses 0.5 or 0.4 depending on code.
    # Let's use 0.5 as a safe default.
    
    identity_id = None
    if name != "Unknown" and similarity > 0.5:
        identity_id = db.get_identity_id_by_name(name)
    else:
        name = "Unknown"
        
    # 2. Check for active sighting
    sighting_id = None
    status = "created"
    
    if identity_id:
        sighting_id = db.get_active_sighting(camera_name, identity_id, threshold_seconds=15)
        
        if sighting_id:
            # 3. Merge
            db.update_sighting_end_time(sighting_id)
            status = "merged"
            db.update_last_seen(name, camera_name) # Heartbeat location
            
    # If not merged, create new
    if not sighting_id:
        sighting_id = db.add_sighting(
            image_path="api_vector_upload", 
            embedding=embedding, 
            identity_id=identity_id, 
            camera=camera_name
        )
        if name != "Unknown":
            db.update_last_seen(name, camera_name)

    # Return embedding too
    embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
    
    return {
        "status": status,
        "name": name,
        "sighting_id": sighting_id,
        "confidence": float(similarity) if similarity else 0.0,
        "embedding": embedding_list
    }

# --- Endpoints ---

@app.post("/detect_face", response_model=DetectResponse)
async def detect_face(camera: str, file: UploadFile = File(...)):
    """
    Upload an image. Detect face. Identify. Tracking.
    """
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image data")
        
    # Detect
    faces = face_app.get(img)
    
    if not faces:
        return {
            "status": "no_face_detected",
            "name": "None",
            "sighting_id": 0,
            "message": "No face found in image"
        }
        
    # Pick largest face
    # Sort by area (det is [x1, y1, x2, y2])
    faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
    face = faces[0]
    
    # Process
    result = process_sighting(face.embedding, camera)
    
    # If detecting from image, we might want to save the crop?
    # For now, respecting "don't pollute" instruction, we stick to minimal logic.
    # Future: Save crop to disk and update `image_path` in DB.
    
    return result

@app.post("/recognize_face", response_model=DetectResponse)
async def recognize_face(request: RecognizeRequest):
    """
    Submit a vector. Identify. Tracking.
    """
    embedding_np = np.array(request.embedding, dtype=np.float32)
    
    if embedding_np.shape != (512,):
        raise HTTPException(status_code=400, detail=f"Invalid embedding shape: {embedding_np.shape}. Expected 512.")
        
    result = process_sighting(embedding_np, request.camera)
    return result

@app.get("/health")
def health():
    return {"status": "ok"}
