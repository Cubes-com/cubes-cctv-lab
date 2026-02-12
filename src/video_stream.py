import cv2
import threading
import time

class LatestFrameReader:
    def __init__(self, src, reconnect_interval=5):
        self.src = src
        self.reconnect_interval = reconnect_interval
        self.cap = None
        self.frame = None
        self.ret = False
        self.stopped = False
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        
        # Start background thread
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while not self.stopped:
            if self.cap is None or not self.cap.isOpened():
                print(f"Connecting to {self.src}...")
                self.cap = cv2.VideoCapture(self.src)
                if not self.cap.isOpened():
                    print(f"Failed to open {self.src}. Retrying in {self.reconnect_interval}s...")
                    time.sleep(self.reconnect_interval)
                    continue
                print(f"Connected to {self.src}")

            # Read frame
            ret, frame = self.cap.read()
            
            with self.lock:
                if ret:
                    self.ret = True
                    self.frame = frame
                    self.condition.notify_all() # Notify valid frame
                else:
                    self.ret = False
                    print(f"Stream {self.src} ended or failed. Reconnecting...")
                    self.cap.release()
                    self.cap = None
                    time.sleep(1) # Prevent tight loop on failure

    def read(self):
        """
        Returns the latest frame. 
        Block until at least one frame is available if none yet.:w
        """
        with self.lock:
            # If no frame yet, wait
            if self.frame is None:
                self.condition.wait(timeout=5.0)
            
            return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.thread.join()
        if self.cap:
            self.cap.release()

    def isOpened(self):
        return True # Wrapper always "open" as it manages connection internally
