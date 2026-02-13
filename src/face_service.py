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
def process_sighting(embedding, camera_name, record=True, face_img=None, full_img=None, face_bbox=None):
    """
    Common logic for handling a face embedding:
    1. Identify person.
    2. Check for active sighting.
    3. Merge or Create (if record=True).
    4. Save Image (if record=True and image provided).
    """
    # 1. Identify
    # get_best_match returns (name, similarity)
    name, similarity = db.get_best_match_with_score(embedding)
    
    identity_id = None
    if name != "Unknown" and similarity > 0.5:
        identity_id = db.get_identity_id_by_name(name)
    else:
        name = "Unknown"
    
    sighting_id = None
    status = "identified"
    
    if record:
        status = "created"
        # 2. Check for active sighting
        if identity_id:
            sighting_id = db.get_active_sighting(camera_name, identity_id, threshold_seconds=15)
            
            if sighting_id:
                # 3. Merge
                db.update_sighting_end_time(sighting_id)
                status = "merged"
                db.update_last_seen(name, camera_name) # Heartbeat location
                
        # If not merged, create new
        if not sighting_id:
            # 4. Save Image
            image_path = "api_vector_upload" # Default if no image
            bbox_json = None
            
            if face_img is not None:
                # Generate unique filename
                timestamp = int(datetime.datetime.now().timestamp() * 1000)
                filename = f"{timestamp}_{camera_name}_api.jpg"
                save_dir = "data/sightings"
                
                # Ensure dir exists (it is mounted, but safety first)
                os.makedirs(save_dir, exist_ok=True)
                
                filepath = os.path.join(save_dir, filename)
                
                # Save just the face crop for now (consistent with analyse_rtsp logic for knowns)
                # Or should we save full image if available? 
                # analyse_rtsp saves person_crop. 
                # Here face_img IS the crop if detected via face analysis? NO, face_img is likely the cropped alignment or we need to crop it.
                # Actually, in _handle_image we have `img` (full) and `face` object.
                # Let's pass the crop as `face_img`.
                
                cv2.imwrite(filepath, face_img)
                image_path = filepath
                
                # Bbox
                if face_bbox is not None and full_img is not None:
                    h, w, _ = full_img.shape
                    # normalize
                    b = face_bbox
                    bbox_json = [
                        float(round(b[0]/w, 4)), float(round(b[1]/h, 4)),
                        float(round(b[2]/w, 4)), float(round(b[3]/h, 4))
                    ]

            sighting_id = db.add_sighting(
                image_path=image_path, 
                embedding=embedding, 
                identity_id=identity_id, 
                camera=camera_name,
                bbox=bbox_json
            )
            if name != "Unknown":
                db.update_last_seen(name, camera_name)

    # Return embedding too
    embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
    
    return {
        "status": status,
        "name": name,
        "sighting_id": sighting_id if sighting_id else 0,
        "confidence": float(similarity) if similarity else 0.0,
        "embedding": embedding_list
    }

# --- Endpoints ---

@app.post("/record_face_image", response_model=DetectResponse)
async def record_face_image(camera: str, file: UploadFile = File(...)):
    """
    Upload an image. Detect face. Identify. Tracking.
    """
    return await _handle_image(camera, file, record=True)

@app.post("/identify_face_image", response_model=DetectResponse)
async def identify_face_image(camera: str, file: UploadFile = File(...)):
    """
    [Read-Only] Upload an image. Detect face. Identify. NO Tracking.
    """
    return await _handle_image(camera, file, record=False)

@app.post("/record_face_vector", response_model=DetectResponse)
async def record_face_vector(request: RecognizeRequest):
    """
    Submit a vector. Identify. Tracking.
    """
    return _handle_vector(request, record=True)

@app.post("/identify_face_vector", response_model=DetectResponse)
async def identify_face_vector(request: RecognizeRequest):
    """
    [Read-Only] Submit a vector. Identify. NO Tracking.
    """
    return _handle_vector(request, record=False)

# --- Internal Handlers ---

async def _handle_image(camera: str, file: UploadFile, record: bool):
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
    faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
    face = faces[0]
    
    # Crop Face for saving
    # InsightFace bbox is [x1, y1, x2, y2]
    b = list(map(int, face.bbox))
    # Clip to bounds
    h, w, _ = img.shape
    b[0] = max(0, b[0]); b[1] = max(0, b[1])
    b[2] = min(w, b[2]); b[3] = min(h, b[3])
    
    face_crop = img[b[1]:b[3], b[0]:b[2]]
    
    return process_sighting(face.embedding, camera, record=record, face_img=face_crop, full_img=img, face_bbox=face.bbox)

def _handle_vector(request: RecognizeRequest, record: bool):
    embedding_np = np.array(request.embedding, dtype=np.float32)
    
    if embedding_np.shape != (512,):
        raise HTTPException(status_code=400, detail=f"Invalid embedding shape: {embedding_np.shape}. Expected 512.")
        
    return process_sighting(embedding_np, request.camera, record=record)
@app.get("/health")
def health():
    return {"status": "ok"}
