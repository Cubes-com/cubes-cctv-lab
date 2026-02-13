import cv2
import time
import os
# Suppress C++ logging from MediaPipe/TensorFlow
os.environ["GLOG_minloglevel"] = "2"

import argparse
import datetime
import numpy as np
import onnxruntime as ort
import supervision as sv
import insightface
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
from identity_db import IdentityDB
from video_stream import LatestFrameReader
import collections

class StatsTracker:
    def __init__(self, window_seconds=600):
        self.window_seconds = window_seconds
        self.processed_timestamps = collections.deque()
        self.skipped_timestamps = collections.deque()
        self.detection_counts = collections.deque() # Stores (timestamp, count) tuples
        
    def add_processed(self):
        self.processed_timestamps.append(time.time())
        
    def add_skipped(self):
        self.skipped_timestamps.append(time.time())
        
    def add_detection(self, count):
        if count > 0:
            self.detection_counts.append((time.time(), count))
            
    def prune(self):
        now = time.time()
        cutoff = now - self.window_seconds
        
        while self.processed_timestamps and self.processed_timestamps[0] < cutoff:
            self.processed_timestamps.popleft()
            
        while self.skipped_timestamps and self.skipped_timestamps[0] < cutoff:
            self.skipped_timestamps.popleft()
            
        while self.detection_counts and self.detection_counts[0][0] < cutoff:
            self.detection_counts.popleft()
            
    def get_stats(self):
        # We prune before getting stats to ensure accuracy
        self.prune()
        
        processed = len(self.processed_timestamps)
        skipped = len(self.skipped_timestamps)
        detected = sum(count for _, count in self.detection_counts)
        
        return processed, skipped, detected

def preprocess(frame, input_shape):
    # Resize to input_shape
    img = cv2.resize(frame, input_shape)
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize to 0-1, transpose to CHW, add batch dim
    img_data = img.transpose(2, 0, 1).astype('float32') / 255.0
    img_data = np.expand_dims(img_data, axis=0)
    return img_data

# ... (skip to main loop content removal) ...
# I cannot do two disjoint edits in one replace_file_content unless I use multi_replace.
# I will use multi_replace_file_content.

def postprocess(output, input_shape, original_shape):
    # output shape is (1, 84, 8400)
    predictions = np.squeeze(output).T  # (8400, 84)

    SCORE_THRESH = 0.2  # start here
    NMS_THRESH = 0.5    # start here

    # DEBUG one-time
    print("boxes min/max:", float(np.min(predictions[:, :4])), float(np.max(predictions[:, :4])))
    print("class min/max:", float(np.min(predictions[:, 4:])), float(np.max(predictions[:, 4:])))

    scores = np.max(predictions[:, 4:], axis=1)
    keep = scores > SCORE_THRESH
    predictions = predictions[keep, :]
    scores = scores[keep]

    if len(scores) == 0:
        return sv.Detections.empty()

    print("max score:", float(scores.max()), "kept:", int((scores > SCORE_THRESH).sum()))

    class_ids = np.argmax(predictions[:, 4:], axis=1)
    boxes = predictions[:, :4]
    
    input_h, input_w = input_shape
    orig_h, orig_w = original_shape
    
    x_factor = orig_w / input_w
    y_factor = orig_h / input_h

    boxes_xyxy = []
    boxes_xywh = []
    
    for box in boxes:
        cx, cy, w, h = box
        
        # Scale back to original image
        cx *= x_factor
        cy *= y_factor
        w *= x_factor
        h *= y_factor
        
        x1 = int(cx - w/2)
        y1 = int(cy - h/2)
        x2 = int(cx + w/2)
        y2 = int(cy + h/2)
        
        boxes_xyxy.append([x1, y1, x2, y2])
        boxes_xywh.append([x1, y1, int(w), int(h)])
        
    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), SCORE_THRESH, NMS_THRESH)
    if len(indices) == 0:
        return sv.Detections.empty()
    
    indices = indices.flatten()

    return sv.Detections(
        xyxy=np.array(boxes_xyxy)[indices],
        confidence=scores[indices],
        class_id=class_ids[indices]
    )

def main():
    parser = argparse.ArgumentParser(description="Run CCTV Analysis")
    parser.add_argument("--name", required=True, help="Camera Name")
    parser.add_argument("--url", required=True, help="RTSP URL")
    args = parser.parse_args()

    rtsp_url = args.url
    camera_name = args.name
    print(f"Starting analysis for {camera_name} at {rtsp_url}")

    model_path = "yolov8n.onnx"
    gesture_model_path = "gesture_recognizer.task"
    
    # 1. Load YOLO
    print(f"Loading YOLO model from {model_path}...")
    try:
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
        return

    print("ORT providers:", ort.get_available_providers())
    print("Model inputs:")
    for i in session.get_inputs():
        print("  -", i.name, i.shape, i.type)

    print("Model outputs:")
    for o in session.get_outputs():
        print("  -", o.name, o.shape, o.type)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape = (640, 640) 

    # 2. Init Face Analysis (InsightFace)
    print("Initializing Face Analysis (w/ partial SCRFD)...")
    # buffalo_s uses DET_500M (SCRFD/RetinaFace)
    # Increasing det_thresh to 0.65 to avoid back-of-head detections
    face_app = insightface.app.FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.65)
    
    # 3. Init Gesture Recognizer (MediaPipe) - DISABLED
    # print(f"Initializing Gesture Recognizer from {gesture_model_path}...")
    # BaseOptions = python.BaseOptions
    # GestureRecognizer = vision.GestureRecognizer
    # GestureRecognizerOptions = vision.GestureRecognizerOptions
    # VisionRunningMode = vision.RunningMode
    #
    # # Create a gesture recognizer instance with the image mode:
    # options = GestureRecognizerOptions(
    #     base_options=BaseOptions(model_asset_path=gesture_model_path),
    #     running_mode=VisionRunningMode.IMAGE,
    #     num_hands=2,
    #     min_hand_detection_confidence=0.5,
    #     min_hand_presence_confidence=0.5,
    #     min_tracking_confidence=0.5
    # )
    # recognizer = GestureRecognizer.create_from_options(options)

    # 4. Init Identity DB
    db = IdentityDB()
    
    # 5. Stats Tracker
    stats = StatsTracker(window_seconds=600)
    last_stats_update = time.time()

    # Initialize Tracker
    tracker = sv.ByteTrack()

    # Track ID -> Info cache
    # {
    #   track_id: {
    #       "name": str, 
    #       "last_face_check": timestamp, 
    #       "last_gesture_check": timestamp,
    #       "current_gesture": str
    #   }
    # }
    track_info = {}

    print(f"Connecting to {rtsp_url}...")
    # cap = cv2.VideoCapture(rtsp_url)
    cap = LatestFrameReader(rtsp_url)
    time.sleep(2) # Allow connection time

    # if not cap.isOpened():
    #     print(f"Error: Could not open stream {rtsp_url}")
    #     return

    print(f"Successfully connected to {rtsp_url}")

    prev_time = time.time()
    frame_count = 0
    last_frame_id = -1

    try:
        while True:
            ret, frame, frame_id = cap.read()
            if not ret:
                print("No frame yet. Waiting...")
                stats.add_skipped()
                time.sleep(0.01)
                continue                
            # Ignore duplicate frames (consumer faster than producer)
            if frame_id == last_frame_id:
                time.sleep(0.01)
                continue

            # Calculate skipped frames
            if last_frame_id != -1:
                skipped_frames = frame_id - last_frame_id - 1
                if skipped_frames > 0:
                    # Optimized addition for large skips
                    for _ in range(skipped_frames):
                        stats.add_skipped()
            
            last_frame_id = frame_id
            
            stats.add_processed()
            frame_count += 1
        

            # --- Periodic Stats Update (Every 30s) ---
            if time.time() - last_stats_update > 30:
                p, s, d = stats.get_stats()
                db.update_camera_stats(camera_name, p, s, d)
                last_stats_update = time.time()
            
            # --- YOLO Inference ---
            img_data = preprocess(frame, input_shape)
            outputs = session.run([output_name], {input_name: img_data})
            detections = postprocess(outputs[0], input_shape, frame.shape[:2])
            
            # Filter for Person (class_id == 0)
            detections = detections[detections.class_id == 0]
            
            stats.add_detection(len(detections))

            # Tracking
            detections = tracker.update_with_detections(detections)
            
            # --- Person Analysis Loop ---
            now = time.time()
            
            if time.time() - last_stats_update > 5:
                out0 = outputs[0]
                print("YOLO output shape:", out0.shape, "dtype:", out0.dtype,
                    "min:", float(np.min(out0)), "max:", float(np.max(out0)))

            for i, track_id in enumerate(detections.tracker_id):
                if track_id not in track_info:
                    track_info[track_id] = {
                        "name": "Unknown", 
                        "last_face_check": 0,
                        "last_gesture_check": 0,
                        "current_gesture": None
                    }
                
                info = track_info[track_id]
                
                # Get Bounding Box
                bbox = detections.xyxy[i]
                x1, y1, x2, y2 = map(int, bbox)
                h, w, _ = frame.shape
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w, x2); y2 = min(h, y2)
                
                # Only process if box has valid size
                if x2 <= x1 or y2 <= y1:
                    continue
            
            # DEBUG: Save one frame to check what we are seeing
                # DEBUG: Save one frame to check what we are seeing
                if camera_name == "2ndfloorlift" and not os.path.exists("data/debug_2ndfloorlift.jpg"):
                    cv2.imwrite("data/debug_2ndfloorlift.jpg", frame)
                    print("DEBUG: Saved data/debug_2ndfloorlift.jpg")
                    
                person_crop = frame[y1:y2, x1:x2]
                
                # 1. Face Recognition Logic
                
                # A) If Unknown, try to identify
                if info["name"] == "Unknown":
                    if (now - info["last_face_check"] > 1.0):
                        try:
                            faces = face_app.get(person_crop)
                            if len(faces) > 0:
                                face = faces[0] 
                                
                                # Extra strict check just in case
                                if face.det_score < 0.65:
                                    continue
                                    
                                matched_name = db.get_best_match(face.embedding)
                                
                                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                filepath = None
                                
                                if matched_name:
                                    info["name"] = matched_name
                                    print(f"IDENTIFIED: Track ID {track_id} is {matched_name}")
                                    
                                    # Update Location in DB (Immediate)
                                    db.update_last_seen(matched_name, camera_name)
                                    info["last_db_update"] = now
                                    
                                    # Save Known Sighting (Throttled)
                                    identity_id = db.get_identity_id_by_name(matched_name)
                                    if identity_id:
                                        filename = f"{timestamp}_track{track_id}_{matched_name}.jpg"
                                        filepath = os.path.join("data/sightings", filename)
                                        
                                        # Normalize BBox (relative to person crop)
                                        fx1, fy1, fx2, fy2 = map(int, face.bbox)
                                        ch, cw, _ = person_crop.shape
                                        # Ensure within bounds
                                        fx1 = max(0, fx1); fy1 = max(0, fy1)
                                        fx2 = min(cw, fx2); fy2 = min(ch, fy2)
                                        
                                        norm_bbox = [
                                            round(fx1 / cw, 4),
                                            round(fy1 / ch, 4),
                                            round(fx2 / cw, 4),
                                            round(fy2 / ch, 4)
                                        ]
                                        
                                        # Save CLEAN image (no drawing)
                                        cv2.imwrite(filepath, person_crop)
                                        
                                        db.add_sighting(filepath, face.embedding, identity_id=identity_id, is_permanent=False, camera=camera_name, bbox=norm_bbox)
                                        print(f"LOG: Saved known sighting to {filepath} with bbox {norm_bbox}")

                                else:
                                    # Still Unknown - Save Sighting (Throttled)
                                    last_saved = info.get("last_sighting_saved", 0)
                                    if (now - last_saved > 10.0):
                                        filename = f"{timestamp}_{camera_name}_track{track_id}.jpg"
                                        filepath = os.path.join("data/sightings", filename)
                                        
                                        # Normalize BBox
                                        fx1, fy1, fx2, fy2 = map(int, face.bbox)
                                        ch, cw, _ = person_crop.shape
                                        fx1 = max(0, fx1); fy1 = max(0, fy1)
                                        fx2 = min(cw, fx2); fy2 = min(ch, fy2)
                                        
                                        norm_bbox = [
                                            round(fx1 / cw, 4),
                                            round(fy1 / ch, 4),
                                            round(fx2 / cw, 4),
                                            round(fy2 / ch, 4)
                                        ]
                                        
                                        # Save CLEAN image
                                        cv2.imwrite(filepath, person_crop)
                                        
                                        db.add_sighting(filepath, face.embedding, is_permanent=False, camera=camera_name, bbox=norm_bbox)
                                        print(f"SIGHTING: Saved unknown sighting to {filepath} with bbox {norm_bbox}")
                                        
                                        info["last_sighting_saved"] = now

                            info["last_face_check"] = now
                        except Exception as e:
                            print(f"Face check error: {e}")
                            pass
                
                # B) If Known, update location heartbeat and re-verify
                else:
                    # Heartbeat: Update location every 15s to keep "Last Seen" fresh
                    last_update = info.get("last_db_update", 0)
                    if (now - last_update > 15.0):
                        # Re-verify? Or just update?
                        # Let's just update for now to solve the specific "stale timestamp" issue.
                        # Re-verifying every 15s is expensive if many people.
                        # Let's check face every 60s maybe?
                        # User priority is "regularly updated". 
                        
                        try:
                            db.update_last_seen(info["name"], camera_name)
                            info["last_db_update"] = now
                            # print(f"HEARTBEAT: Updated location for {info['name']}") 
                        except Exception as e:
                            print(f"Heartbeat error: {e}")
                            
                    # Periodic Re-Verification (e.g. every 60s) to ensure they didn't swap?
                    # ByteTrack is pretty good, but let's be safe.
                    if (now - info["last_face_check"] > 60.0):
                         # Reset to Unknown to force re-check next frame?
                         # Or check manually here.
                         pass
                
                # 2. Gesture Recognition (DISABLED for stability debugging)
                # if (now - info["last_gesture_check"] > 0.5):
                #     try:
                #         # Convert to MP Image (RGB)
                #         # OpenCV is BGR, MediaPipe needs RGB
                #         rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                #         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
                #         
                #         gesture_result = recognizer.recognize(mp_image)
                #         
                #         found_gesture = None
                #         if gesture_result.gestures:
                #             # gesture_result.gestures is a list of lists (one list per hand)
                #             # Let's take the first gesture from the first hand
                #             top_gesture = gesture_result.gestures[0][0]
                #             if top_gesture.category_name != "None":
                #                 found_gesture = top_gesture.category_name
                #         
                #         info["current_gesture"] = found_gesture
                #         
                #         if found_gesture:
                #              print(f"GESTURE: Person #{track_id} ({info['name']}) -> {found_gesture}")
                #         
                #         info["last_gesture_check"] = now
                #     except Exception:
                #          pass

            # Prepare Log Output
            # Log detections
            for track_id in detections.tracker_id:
                info = track_info.get(track_id)
                name = info["name"]
                gesture = info["current_gesture"]
                
                # Format: time - camera_name - person (gesture)
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # Clean name if it has ID? User said [person]
                # info["name"] is usually "Jamie" or "Unknown" or "#ID Jamie"?
                # In main loop: info["name"] = name (e.g. "Jamie")
                # But in labels construction: f"#{track_id} {name}"
                # Let's use the raw name.
                
                log_msg = f"{timestamp} - {camera_name} - {name}"
                if gesture:
                    log_msg += f" ({gesture})"
                    
                # Throttle log to once every 2 seconds per person
                last_log = info.get("last_log_time", 0)
                if (now - last_log) > 2.0:
                    print(log_msg)
                    info["last_log_time"] = now

            frame_count += 1
            curr_time = time.time()
            elapsed = curr_time - prev_time

            if elapsed >= 1.0:
                # fps = frame_count / elapsed
                # print(f"FPS: {fps:.2f} ...") 
                frame_count = 0
                prev_time = curr_time

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        db.close()

if __name__ == "__main__":
    main()
