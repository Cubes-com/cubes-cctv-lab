import urllib.request
import os

# Correct URL for v8.2.0 release assets
url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx"
output = "yolov8n.onnx"

try:
    if not os.path.exists(output):
        print(f"Downloading {url}...")
        # Add User-Agent header as GitHub might block python-urllib
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        
        urllib.request.urlretrieve(url, output)
        print("Download complete.")
    else:
        print("Model already exists.")
except Exception as e:
    print(f"Failed: {e}")
