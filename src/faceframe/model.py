"""Location of the official MediaPipe face detection model. No heavy imports."""

from pathlib import Path

# BlazeFace short range, from the MediaPipe Face Detector docs. Intended for
# faces within roughly 2 m of the camera, i.e. typical webcam distance.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
# Project-root/models/... (assumes an editable install: `pip install -e .`)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "blaze_face_short_range.tflite"
