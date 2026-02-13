import os
import psycopg2
import numpy as np
import datetime
import io
import time

class IdentityDB:
    def __init__(self):
        self.db_host = os.environ.get("DB_HOST", "localhost")
        self.db_name = os.environ.get("DB_NAME", "cctv")
        self.db_user = os.environ.get("DB_USER", "user")
        self.db_pass = os.environ.get("DB_PASS", "password")
        self.match_threshold = float(os.environ.get("FACE_MATCH_THRESHOLD", "0.6"))
        self.conn = None
        self.connect()
        self.create_table()

    def connect(self):
        while not self.conn:
            try:
                self.conn = psycopg2.connect(
                    host=self.db_host,
                    database=self.db_name,
                    user=self.db_user,
                    password=self.db_pass
                )
                print("Connected to PostgreSQL")
            except Exception as e:
                print(f"Database connection failed: {e}. Retrying in 5s...")
                time.sleep(5)

    def create_table(self):
        with self.conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS identities (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    embedding BYTEA NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sightings (
                    id SERIAL PRIMARY KEY,
                    image_path TEXT NOT NULL,
                    embedding BYTEA NOT NULL,
                    identity_id INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_permanent BOOLEAN DEFAULT FALSE,
                    camera TEXT,
                    bbox JSONB,
                    FOREIGN KEY(identity_id) REFERENCES identities(id)
                )
            """)
            # Migration: Add is_permanent if it doesn't exist
            cursor.execute("""
                DO $$ 
                BEGIN 
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='sightings' AND column_name='is_permanent') THEN
                        ALTER TABLE sightings ADD COLUMN is_permanent BOOLEAN DEFAULT FALSE;
                        -- Assume existing named sightings are permanent (manually assigned/uploaded)
                        UPDATE sightings SET is_permanent = TRUE WHERE identity_id IS NOT NULL;
                    END IF; 
                    
                    -- Migration: Add camera if it doesn't exist
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='sightings' AND column_name='camera') THEN
                        ALTER TABLE sightings ADD COLUMN camera TEXT;
                    END IF;

                    -- Migration: Add bbox if it doesn't exist
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='sightings' AND column_name='bbox') THEN
                        ALTER TABLE sightings ADD COLUMN bbox JSONB;
                    END IF;

                    -- Migration: Add end_timestamp if it doesn't exist
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='sightings' AND column_name='end_timestamp') THEN
                        ALTER TABLE sightings ADD COLUMN end_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
                        -- Backfill existing rows
                        UPDATE sightings SET end_timestamp = timestamp WHERE end_timestamp IS NULL;
                    END IF;
                END $$;
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS person_location (
                    name TEXT PRIMARY KEY,
                    camera TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS person_location_log (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    camera TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS camera_stats (
                    camera TEXT PRIMARY KEY,
                    frames_processed INTEGER DEFAULT 0,
                    frames_skipped INTEGER DEFAULT 0,
                    people_detected INTEGER DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        self.conn.commit()

    def update_last_seen(self, name, camera, timestamp=None):
        if timestamp is None:
            timestamp = datetime.datetime.now()
            
        with self.conn.cursor() as cursor:
            # 1. Update latest location (Upsert)
            cursor.execute("""
                INSERT INTO person_location (name, camera, timestamp) 
                VALUES (%s, %s, %s)
                ON CONFLICT(name) DO UPDATE SET 
                camera=EXCLUDED.camera, 
                timestamp=EXCLUDED.timestamp
            """, (name, camera, timestamp))
            
            # 2. Add to log history
            # Only log if it's a "new" event (different from last log? or just always log?)
            # To be robust, let's always log. We can filter later if needed.
            cursor.execute("""
                INSERT INTO person_location_log (name, camera, timestamp)
                VALUES (%s, %s, %s)
            """, (name, camera, timestamp))
            
        self.conn.commit()

    def save_identity(self, name, embedding):
        embedding_bytes = embedding.astype(np.float32).tobytes()
        with self.conn.cursor() as cursor:
            cursor.execute("INSERT INTO identities (name, embedding) VALUES (%s, %s)", (name, embedding_bytes))
        self.conn.commit()
        print(f"Saved identity: {name}")

    def delete_identity(self, name):
        """
        Deletes a person, their sightings (DB + files), and location history.
        Returns list of file paths that were deleted.
        """
        deleted_files = []
        with self.conn.cursor() as cursor:
            # Get ID first
            cursor.execute("SELECT id FROM identities WHERE name = %s", (name,))
            res = cursor.fetchone()
            if not res: return []
            identity_id = res[0]
            
            # 1. Get all sighting file paths to delete
            cursor.execute("SELECT image_path FROM sightings WHERE identity_id = %s", (identity_id,))
            rows = cursor.fetchall()
            for row in rows:
                if row[0]:
                    deleted_files.append(row[0])
            
            # 2. Delete sightings from DB
            cursor.execute("DELETE FROM sightings WHERE identity_id = %s", (identity_id,))
            
            # 3. Delete identity
            cursor.execute("DELETE FROM identities WHERE id = %s", (identity_id,))
            
            # 4. Clean location data
            cursor.execute("DELETE FROM person_location WHERE name = %s", (name,))
            cursor.execute("DELETE FROM person_location_log WHERE name = %s", (name,))
            
        self.conn.commit()
        return deleted_files

    def update_camera_stats(self, camera, processed, skipped, detected):
        with self.conn.cursor() as cursor:
            # Upsert stats
            cursor.execute("""
                INSERT INTO camera_stats (camera, frames_processed, frames_skipped, people_detected, updated_at)
                VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (camera) DO UPDATE SET
                    frames_processed = EXCLUDED.frames_processed,
                    frames_skipped = EXCLUDED.frames_skipped,
                    people_detected = EXCLUDED.people_detected,
                    updated_at = CURRENT_TIMESTAMP
            """, (camera, processed, skipped, detected))
        self.conn.commit()

    def get_all_camera_stats(self):
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT camera, frames_processed, frames_skipped, people_detected, updated_at FROM camera_stats ORDER BY camera ASC")
            return cursor.fetchall()


    def get_known_faces(self):
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT name, embedding FROM identities")
            rows = cursor.fetchall()
        
        known_faces = []
        for name, emb_bytes in rows:
            # Postgres returns memoryview or bytes for BYTEA
            embedding = np.frombuffer(emb_bytes, dtype=np.float32)
            known_faces.append((name, embedding))
            
        return known_faces

    def get_best_match(self, target_embedding, threshold=None):
        name, score = self.get_best_match_with_score(target_embedding, threshold)
        return name

    def get_best_match_with_score(self, target_embedding, threshold=None):
        if threshold is None:
            threshold = self.match_threshold
        
        # Optimization: Check if any known faces exist first?
        # The logic below handles it (rows will be empty)
        
        norm = np.linalg.norm(target_embedding)
        if norm > 0:
            target_embedding = target_embedding / norm

        best_score = -1.0
        best_name = None

        with self.conn.cursor() as cursor:
            # Base identities
            cursor.execute("SELECT name, embedding FROM identities")
            rows = cursor.fetchall()
            
            # Assigned sightings (Only PERMANENT/CONFIRMED ones)
            # Optimization: Cache this?
            cursor.execute("""
                SELECT i.name, s.embedding 
                FROM sightings s 
                JOIN identities i ON s.identity_id = i.id
                WHERE s.is_permanent = TRUE
            """)
            rows.extend(cursor.fetchall())
        
        for name, emb_bytes in rows:
            db_embedding = np.frombuffer(emb_bytes, dtype=np.float32)
            score = np.dot(target_embedding, db_embedding)
            if score > best_score:
                best_score = score
                best_name = name
        
        if best_score > threshold:
            return best_name, best_score
        
        return None, best_score

    def add_sighting(self, image_path, embedding, identity_id=None, is_permanent=False, camera=None, bbox=None):
        embedding_bytes = embedding.astype(np.float32).tobytes()
        
        # Ensure bbox is a list or json string if provided
        import json
        bbox_json = json.dumps(bbox) if bbox else None

        sighting_id = None
        with self.conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO sightings (image_path, embedding, identity_id, is_permanent, camera, bbox, end_timestamp) VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP) RETURNING id", 
                (image_path, embedding_bytes, identity_id, is_permanent, camera, bbox_json)
            )
            sighting_id = cursor.fetchone()[0]
        self.conn.commit()
        return sighting_id

    def update_sighting_end_time(self, sighting_id):
        """Updates the end_timestamp of a sighting to now."""
        with self.conn.cursor() as cursor:
            cursor.execute("UPDATE sightings SET end_timestamp = CURRENT_TIMESTAMP WHERE id = %s", (sighting_id,))
        self.conn.commit()

    def get_active_sighting(self, camera, identity_id, threshold_seconds=10):
        """
        Finds a recent sighting for the given identity on the given camera.
        Used to merge consecutive detections into a single sighting.
        """
        with self.conn.cursor() as cursor:
            # Check for a sighting where end_timestamp is within the last X seconds
            cursor.execute("""
                SELECT id 
                FROM sightings 
                WHERE camera = %s 
                  AND identity_id = %s 
                  AND end_timestamp > (CURRENT_TIMESTAMP - INTERVAL '%s seconds')
                ORDER BY end_timestamp DESC 
                LIMIT 1
            """, (camera, identity_id, threshold_seconds))
            res = cursor.fetchone()
        return res[0] if res else None

    def add_sighting_for_identity(self, identity_id, image_path, embedding, camera=None, bbox=None):
        # Manually added, so is_permanent=True
        self.add_sighting(image_path, embedding, identity_id, is_permanent=True, camera=camera, bbox=bbox)
    
    def get_unassigned_sightings(self):
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT id, image_path, timestamp, camera, bbox FROM sightings WHERE identity_id IS NULL ORDER BY timestamp DESC")
            return cursor.fetchall()
            
    def get_sightings_for_identity(self, identity_id, limit=50):
        with self.conn.cursor() as cursor:
            # Union query:
            # 1. All UNCONFIRMED sightings for this identity (no limit)
            # 2. Recent sightings (confirmed or not) limited by 'limit'
            # This ensures we always see "action items" (unconfirmed) even if they are old.
            cursor.execute("""
                (SELECT id, image_path, timestamp, is_permanent, camera, bbox 
                 FROM sightings 
                 WHERE identity_id = %s AND is_permanent = FALSE)
                UNION
                (SELECT id, image_path, timestamp, is_permanent, camera, bbox 
                 FROM sightings 
                 WHERE identity_id = %s 
                 ORDER BY timestamp DESC 
                 LIMIT %s)
                ORDER BY timestamp DESC
            """, (identity_id, identity_id, limit))
            return cursor.fetchall()
    
    def get_identity_id_by_name(self, name):
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT id FROM identities WHERE name = %s", (name,))
            res = cursor.fetchone()
        return res[0] if res else None

    # --- New Query Methods ---
    def get_all_identities(self):
        """
        Returns list of {name, last_seen_camera, last_seen_timestamp}
        """
        with self.conn.cursor() as cursor:
            # Join with person_location to get last seen info
            cursor.execute("""
                SELECT i.name, pl.camera, pl.timestamp 
                FROM identities i
                LEFT JOIN person_location pl ON i.name = pl.name
                ORDER BY i.name ASC
            """)
            return cursor.fetchall()

    def query_sightings(self, name=None, camera=None, date_str=None, limit=200, order="DESC", min_timestamp=None, max_timestamp=None):
        """
        Flexible sighting query.
        date_str: "today", "yesterday", "YYYY-MM-DD"
        order: "ASC" (First) or "DESC" (Last/Recent)
        min_timestamp, max_timestamp: datetime objects for granular filtering
        """
        query = """
            SELECT s.id, s.image_path, s.timestamp, s.camera, i.name, s.bbox, s.end_timestamp
            FROM sightings s
            LEFT JOIN identities i ON s.identity_id = i.id
            WHERE 1=1
        """
        params = []
        
        if name:
            query += " AND i.name = %s"
            params.append(name)
            
        if camera:
            if isinstance(camera, list):
                # Handle list of cameras (e.g. for split views)
                # Ensure we have valid strings
                valid_cams = [c for c in camera if c]
                if valid_cams:
                    placeholders = ','.join(['%s'] * len(valid_cams))
                    query += f" AND s.camera IN ({placeholders})"
                    params.extend(valid_cams)
            else:
                query += " AND s.camera = %s"
                params.append(camera)
            
        if date_str:
            today = datetime.date.today()
            if date_str == "today":
                start_dt = datetime.datetime.combine(today, datetime.time.min)
                end_dt = datetime.datetime.combine(today, datetime.time.max)
            elif date_str == "yesterday":
                yesterday = today - datetime.timedelta(days=1)
                start_dt = datetime.datetime.combine(yesterday, datetime.time.min)
                end_dt = datetime.datetime.combine(yesterday, datetime.time.max)
            else:
                try:
                    target_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
                    start_dt = datetime.datetime.combine(target_date, datetime.time.min)
                    end_dt = datetime.datetime.combine(target_date, datetime.time.max)
                except ValueError:
                    start_dt = None
                    end_dt = None
            
            if start_dt and end_dt:
                query += " AND s.timestamp >= %s AND s.timestamp <= %s"
                params.append(start_dt)
                params.append(end_dt)

        if min_timestamp:
            query += " AND s.timestamp >= %s"
            params.append(min_timestamp)
        
        if max_timestamp:
            query += " AND s.timestamp <= %s"
            params.append(max_timestamp)
            
        # ordering
        if order.upper() == "ASC":
            query += " ORDER BY s.timestamp ASC"
        else:
            query += " ORDER BY s.timestamp DESC"
            
        if limit:
            query += " LIMIT %s"
            params.append(limit)
            
        with self.conn.cursor() as cursor:
            cursor.execute(query, tuple(params))
            return cursor.fetchall()

    # --- Robust Cleanup Logic ---
    
    def find_session_window(self, name, camera, timestamp):
        """
        Finds the start and end of a contiguous session where 'name' was at 'camera' around 'timestamp'.
        Returns (start_time, end_time).
        """
        # Heuristic: 
        # Look for contiguous person_location_log entries for (name, camera) around the timestamp.
        # "Contiguous" means gaps between logs are small (e.g. < 60 seconds).
        # OR: Just take a fixed window? No, session is better.
        
        # Implementation:
        # 1. Look backwards from timestamp until camera changes or gap > X.
        # 2. Look forwards from timestamp until camera changes or gap > X.
        
        # Actually, simpler approach for "Ghost" removal:
        # User said: "Remove any sightings for User A at that camera during that time."
        # "From the last camera they were at before that (if any) to the next camera they were at after that (if any)."
        pass
        
        with self.conn.cursor() as cursor:
            # Find previous camera change
            cursor.execute("""
                SELECT timestamp FROM person_location_log 
                WHERE name = %s AND timestamp < %s AND camera != %s
                ORDER BY timestamp DESC LIMIT 1
            """, (name, timestamp, camera))
            prev_row = cursor.fetchone()
            start_time = prev_row[0] if prev_row else datetime.datetime.min
            
            # Find next camera change
            cursor.execute("""
                SELECT timestamp FROM person_location_log 
                WHERE name = %s AND timestamp > %s AND camera != %s
                ORDER BY timestamp ASC LIMIT 1
            """, (name, timestamp, camera))
            next_row = cursor.fetchone()
            end_time = next_row[0] if next_row else datetime.datetime.max
            
        return start_time, end_time

    def cleanup_misassigned_session(self, name, camera, timestamp):
        """
        Removes sightings and history for 'name' at 'camera' within the session surrounding 'timestamp'.
        Used when a sighting is moved AWAY from this person.
        """
        if not name or not camera or not timestamp: return

        start_time, end_time = self.find_session_window(name, camera, timestamp)
        
        with self.conn.cursor() as cursor:
            # 1. Unassign sightings (set to Unknown) within this window at this camera
            # Only unassign if they were AUTO-assigned (is_permanent=FALSE)? 
            # Or even confirmed ones?
            # User said: "Remove any sightings for User A".
            # Probably means unassign them so they go back to training pool or drag them to new user?
            # The specific image being moved is handled by the caller.
            # Here we handle the SIDE EFFECTS (other sightings in the same track/session).
            # If we just moved ONE image, we assume the whole track was wrong.
            cursor.execute("""
                UPDATE sightings SET identity_id = NULL, is_permanent = FALSE
                WHERE identity_id = (SELECT id FROM identities WHERE name = %s)
                AND camera = %s
                AND timestamp >= %s AND timestamp <= %s
            """, (name, camera, start_time, end_time))
            
            # 2. Delete from person_location_log
            cursor.execute("""
                DELETE FROM person_location_log
                WHERE name = %s
                AND camera = %s
                AND timestamp >= %s AND timestamp <= %s
            """, (name, camera, start_time, end_time))
            
        self.conn.commit()
        
        # Update latest location for this person (re-calc from remaining data)
        self.refresh_latest_location(name)


    def create_identity_from_sighting(self, sighting_id, name):
        with self.conn.cursor() as cursor:
            # Get info about the sighting
            cursor.execute("SELECT identity_id, camera, timestamp, embedding FROM sightings WHERE id = %s", (sighting_id,))
            res = cursor.fetchone()
            if not res: return False
            old_identity_id, camera, timestamp, embedding_bytes = res
            
            # Get old identity name if exists
            old_name = None
            if old_identity_id:
                cursor.execute("SELECT name FROM identities WHERE id = %s", (old_identity_id,))
                res_name = cursor.fetchone()
                if res_name: old_name = res_name[0]

            # 1. Create Identity
            cursor.execute("INSERT INTO identities (name, embedding) VALUES (%s, %s) RETURNING id", (name, embedding_bytes))
            new_id = cursor.fetchone()[0]
            
            # 2. Assign Sighting
            cursor.execute("UPDATE sightings SET identity_id = %s, is_permanent = TRUE WHERE id = %s", (new_id, sighting_id))
            
        self.conn.commit()
        
        # 3. Cleanup Old Identity (Ghost removal)
        if old_name and camera:
            self.cleanup_misassigned_session(old_name, camera, timestamp)
            
        # 4. Update New Identity Location
        # Since this is a new identity, we just set their location to this sighting?
        # Or add to log? 
        if camera:
            # Add session availability for new user?
            # For now just update current location.
            self.update_last_seen(name, camera, timestamp)
            
        return True

    def assign_sighting(self, sighting_id, identity_name):
        identity_id = self.get_identity_id_by_name(identity_name)
        if not identity_id: return False
        
        with self.conn.cursor() as cursor:
            # Get sighting info
            cursor.execute("SELECT identity_id, camera, timestamp FROM sightings WHERE id = %s", (sighting_id,))
            res = cursor.fetchone()
            if not res: return False
            old_identity_id, camera, timestamp = res
            
            # Get old identity name
            old_name = None
            if old_identity_id:
                cursor.execute("SELECT name FROM identities WHERE id = %s", (old_identity_id,))
                res_name = cursor.fetchone()
                if res_name: old_name = res_name[0]
            
            # Assign
            cursor.execute("UPDATE sightings SET identity_id = %s, is_permanent = TRUE WHERE id = %s", (identity_id, sighting_id))
            
        self.conn.commit()
        
        # Cleanup Old Identity
        if old_name and camera and old_name != identity_name:
            self.cleanup_misassigned_session(old_name, camera, timestamp)
            
        # Update New Identity
        if camera:
            self.update_last_seen(identity_name, camera, timestamp)
        else:
            self.refresh_latest_location(identity_name)
                    
        return True

    def delete_sighting(self, sighting_id):
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT image_path, identity_id, camera, timestamp FROM sightings WHERE id = %s", (sighting_id,))
            res = cursor.fetchone()
            if not res: return False
            
            image_path, identity_id, camera, timestamp = res
            
            # Get name
            name = None
            if identity_id:
                cursor.execute("SELECT name FROM identities WHERE id = %s", (identity_id,))
                res_name = cursor.fetchone()
                if res_name: name = res_name[0]
            
            # Delete from DB
            cursor.execute("DELETE FROM sightings WHERE id = %s", (sighting_id,))
            
        self.conn.commit()
        
        # Cleanup Session Logic
        # If we delete a sighting, do we assume the whole session was wrong?
        # Maybe? Or maybe just this one frame was bad.
        # User said: "Remove any sightings for User A at that camera during that time."
        # This implies session cleanup.
        if name and camera:
            self.cleanup_misassigned_session(name, camera, timestamp)
        
        # Delete file from disk
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except OSError as e:
                print(f"Error removing file {image_path}: {e}")
                
        return True

    def cleanup_old_logs(self, hours=24):
        cutoff = datetime.datetime.now() - datetime.timedelta(hours=hours)
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT image_path FROM sightings WHERE is_permanent = FALSE AND timestamp < %s", (cutoff,))
            rows = cursor.fetchall()
            
            cursor.execute("DELETE FROM sightings WHERE is_permanent = FALSE AND timestamp < %s", (cutoff,))
            deleted_count = cursor.rowcount
            
        self.conn.commit()
        return rows, deleted_count

    def get_all_embeddings_for_identity(self, name):
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT embedding, id FROM identities WHERE name = %s", (name,))
            res = cursor.fetchone()
            if not res: return []
            
            base_embedding = np.frombuffer(res[0], dtype=np.float32)
            identity_id = res[1]
            
            embeddings = [base_embedding]
            
            cursor.execute("SELECT embedding FROM sightings WHERE identity_id = %s AND is_permanent = TRUE", (identity_id,))
            rows = cursor.fetchall()
            for r in rows:
                embeddings.append(np.frombuffer(r[0], dtype=np.float32))
                
        return embeddings

    def close(self):
        if self.conn:
            self.conn.close()

    def confirm_all_sightings(self, identity_id):
        with self.conn.cursor() as cursor:
            """
            When confirming all, we should probably check if any of them are 'ghosts' from other users?
            No, confirm_all usually assumes the current tentative assignments are correct.
            But we should ensure they have 'is_permanent = TRUE'.
            """
            cursor.execute("UPDATE sightings SET is_permanent = TRUE WHERE identity_id = %s", (identity_id,))
            
            # Also, we should probably update the vectors for training?
            # (Vectors are pulled dynamically in get_best_match / get_all_embeddings, so just setting is_permanent=TRUE enables them)
            
        self.conn.commit()
        
        # Refresh location just in case
        with self.conn.cursor() as cursor:
             cursor.execute("SELECT name FROM identities WHERE id = %s", (identity_id,))
             res = cursor.fetchone()
             if res:
                 self.refresh_latest_location(res[0])

    def get_unconfirmed_count(self, identity_id):
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM sightings WHERE identity_id = %s AND is_permanent = FALSE", (identity_id,))
            res = cursor.fetchone()
        return res[0] if res else 0

    def get_unknown_count(self):
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM sightings WHERE identity_id IS NULL")
            res = cursor.fetchone()
        return res[0] if res else 0
        
    def refresh_latest_location(self, name):
        """
        Recalculates the latest location based on DB state.
        Now uses 'camera' column intelligently.
        """
        if not name: return
        
        with self.conn.cursor() as cursor:
            # 1. Get latest from log
            cursor.execute("SELECT camera, timestamp FROM person_location_log WHERE name = %s ORDER BY timestamp DESC LIMIT 1", (name,))
            res_log = cursor.fetchone()
            
            # 2. Get latest from INDIVIDUAL SIGHTINGS (which might be newer if we just uploaded/assigned)
            # Use 'camera' column now!
            cursor.execute("""
                SELECT s.camera, s.timestamp 
                FROM sightings s
                JOIN identities i ON s.identity_id = i.id
                WHERE i.name = %s AND s.camera IS NOT NULL
                ORDER BY s.timestamp DESC LIMIT 1
            """, (name,))
            res_sighting = cursor.fetchone()
            
            best_cam = None
            best_ts = None
            
            # Compare log vs sighting
            if res_log:
                best_cam = res_log[0]
                best_ts = res_log[1]
                
            if res_sighting:
                # Use 'camera' column and timestamp from sighting
                s_camera = res_sighting[0]
                s_time = res_sighting[1]
                
                if s_camera and (best_ts is None or s_time > best_ts):
                    best_cam = s_camera
                    best_ts = s_time
            
            # Update person_location if we found something
            if best_cam and best_ts:
                # Direct update to View (avoid infinite log loop)
                cursor.execute("""
                    INSERT INTO person_location (name, camera, timestamp) 
                    VALUES (%s, %s, %s)
                    ON CONFLICT(name) DO UPDATE SET 
                    camera=EXCLUDED.camera, 
                    timestamp=EXCLUDED.timestamp
                """, (name, best_cam, best_ts))
        
        self.conn.commit()
