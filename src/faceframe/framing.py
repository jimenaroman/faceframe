"""Framing: decide which part of the frame to show.

Phase 1 only defines the interface and a do-nothing implementation, so the
app's main loop already has the seam where the baseline and improved
framing algorithms will plug in later.
"""

from __future__ import annotations

from typing import Protocol, Sequence

from faceframe.faces import FaceDetection
from faceframe.geometry import BoundingBox, FrameSize, full_frame


class Framer(Protocol):
    def compute_crop(self, detections: Sequence[FaceDetection], size: FrameSize) -> BoundingBox:
        """Return the crop rectangle (in frame pixels) to display."""
        ...


class FullFrameFramer:
    """Phase 1 placeholder: always shows the whole frame, never zooms."""

    def compute_crop(self, detections: Sequence[FaceDetection], size: FrameSize) -> BoundingBox:
        return full_frame(size)
