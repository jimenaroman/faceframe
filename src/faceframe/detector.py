"""Face detection using the MediaPipe Tasks Vision FaceDetector.

MediaPipe types stop at this module's boundary: `detect()` returns our own
`FaceDetection` dataclasses so the rest of the app (and the tests) never
depend on MediaPipe's object model.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from faceframe.faces import FaceDetection, convert_detections
from faceframe.geometry import FrameSize
from faceframe.model import DEFAULT_MODEL_PATH

class ModelLoadError(RuntimeError):
    """The model file exists but MediaPipe could not load it."""


class FaceDetector:
    """Synchronous MediaPipe face detector in VIDEO running mode.

    VIDEO mode (rather than LIVE_STREAM) processes every frame we hand it,
    synchronously, with no callbacks or dropped frames. That keeps the loop
    simple and makes runs on prerecorded videos reproducible.
    """

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH, min_confidence: float = 0.5) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be in [0, 1], got {min_confidence}")
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Face detection model not found at {model_path}.\n"
                "Download it with:  python scripts/download_model.py"
            )
        options = vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            min_detection_confidence=min_confidence,
        )
        try:
            self._detector = vision.FaceDetector.create_from_options(options)
        except ValueError as e:  # MediaPipe raises ValueError for corrupt/invalid models
            raise ModelLoadError(
                f"Could not load {model_path}: {e}. Delete it and re-run "
                "python scripts/download_model.py"
            ) from e
        self._min_confidence = min_confidence
        self._last_timestamp_ms = -1

    def __enter__(self) -> "FaceDetector":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[FaceDetection]:
        if self._detector is None:
            raise RuntimeError("FaceDetector.detect() called after close()")
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # OpenCV is BGR, MediaPipe wants RGB
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # VIDEO mode requires strictly increasing timestamps.
        timestamp_ms = max(timestamp_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        result = self._detector.detect_for_video(image, timestamp_ms)
        return convert_detections(result.detections, FrameSize(width, height), self._min_confidence)

    def close(self) -> None:
        if self._detector is not None:
            self._detector.close()
            self._detector = None
