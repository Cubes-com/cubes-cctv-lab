import insightface
import os

print("Pre-loading InsightFace models...")
# This triggers the download/extract of the model pack
app = insightface.app.FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
print("Model downloaded to", os.path.expanduser('~/.insightface/models'))
