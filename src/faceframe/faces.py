"""Detection data types and the MediaPipe-to-FaceFrame adapter.

Deliberately free of MediaPipe/OpenCV imports: `convert_detections` is
duck-typed, so it is unit-tested with simple fake objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from faceframe.geometry import BoundingBox, FrameSize, clamp_to_frame


@dataclass(frozen=True)
class FaceDetection:
    box: BoundingBox                          # clamped to the frame
    raw_box: BoundingBox                      # as reported (may extend off-frame)
    score: float
    keypoints: tuple[tuple[float, float], ...]  # normalized (x, y), MediaPipe order


def convert_detections(
    raw_detections: Iterable[Any], size: FrameSize, min_score: float
) -> list[FaceDetection]:
    """Turn MediaPipe Detection objects into validated FaceDetections.

    Pure function (duck-typed), so it is tested with fake objects.
    Drops detections that are below `min_score`, have no category,
    or fall entirely outside the frame.
    """
    results: list[FaceDetection] = []
    for det in raw_detections:
        if not det.categories:
            continue
        score = det.categories[0].score
        if score is None or score < min_score:
            continue
        bb = det.bounding_box
        raw_box = BoundingBox(int(bb.origin_x), int(bb.origin_y), int(bb.width), int(bb.height))
        box = clamp_to_frame(raw_box, size)
        if box is None:
            continue
        keypoints = tuple(
            (float(kp.x), float(kp.y))
            for kp in (det.keypoints or [])
            if kp.x is not None and kp.y is not None
        )
        results.append(FaceDetection(box=box, raw_box=raw_box, score=float(score), keypoints=keypoints))
    return results
