"""Drawing only. No detection or framing decisions are made here."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from faceframe.faces import FaceDetection
from faceframe.geometry import BoundingBox, FrameSize, normalized_to_pixel

BOX_COLOR = (0, 220, 0)       # BGR green
KEYPOINT_COLOR = (0, 200, 255)  # BGR orange
CROP_COLOR = (255, 160, 0)    # BGR blue
TEXT_COLOR = (255, 255, 255)
WARN_COLOR = (0, 0, 255)


def draw_detections(frame: np.ndarray, detections: Sequence[FaceDetection]) -> None:
    """Draw face boxes, scores and the 6 BlazeFace keypoints, in place."""
    size = FrameSize(frame.shape[1], frame.shape[0])
    for det in detections:
        b = det.box
        cv2.rectangle(frame, (b.x, b.y), (b.x2 - 1, b.y2 - 1), BOX_COLOR, 2)
        cv2.putText(frame, f"{det.score:.2f}", (b.x, max(b.y - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 1, cv2.LINE_AA)
        for kx, ky in det.keypoints:
            cv2.circle(frame, normalized_to_pixel(kx, ky, size), 3, KEYPOINT_COLOR, -1)


def draw_crop(frame: np.ndarray, crop: BoundingBox) -> None:
    """Outline the crop the framer chose (invisible in Phase 1: it's the full frame)."""
    cv2.rectangle(frame, (crop.x, crop.y), (crop.x2 - 1, crop.y2 - 1), CROP_COLOR, 1)


def draw_status(frame: np.ndarray, num_faces: int, detect_ms: float) -> None:
    if num_faces == 0:
        text, color = "No face detected", WARN_COLOR
    else:
        text, color = f"Faces: {num_faces}", TEXT_COLOR
    cv2.putText(frame, f"{text} | detect {detect_ms:.1f} ms | q/Esc to quit", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
