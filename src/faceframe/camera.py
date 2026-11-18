"""Video capture wrapper. The only module that talks to camera hardware.

Frames are read and handed to the caller; nothing is ever written to disk.
"""

from __future__ import annotations

import time
from typing import Callable, Protocol

import cv2
import numpy as np


class CameraError(RuntimeError):
    """Raised when a video source cannot be opened or stops delivering frames."""


class CaptureLike(Protocol):
    """The subset of cv2.VideoCapture we use. Lets tests inject a fake."""

    def isOpened(self) -> bool: ...
    def read(self) -> tuple[bool, np.ndarray | None]: ...
    def get(self, prop_id: int) -> float: ...
    def release(self) -> None: ...


CaptureFactory = Callable[[int | str], CaptureLike]

_OPEN_HELP = (
    "Could not open video source {source!r}. Check that:\n"
    "  - a camera is connected and not in use by another app (FaceTime, Zoom, ...)\n"
    "  - this terminal/IDE has camera permission\n"
    "    (macOS: System Settings > Privacy & Security > Camera)\n"
    "  - the index is right (try --source 1), or the file path exists"
)


class VideoSource:
    """Context manager around a webcam index or a video file path.

    For live cameras, a few failed reads are tolerated (cameras often drop
    the first frames while warming up or while the OS permission prompt is
    shown). After `max_consecutive_failures` failed reads in a row we raise
    instead of spinning forever on a dead device.
    """

    def __init__(
        self,
        source: int | str = 0,
        capture_factory: CaptureFactory = cv2.VideoCapture,
        max_consecutive_failures: int = 30,
    ) -> None:
        self.source = source
        self.is_live = isinstance(source, int)
        self._factory = capture_factory
        self._max_failures = max_consecutive_failures
        self._capture: CaptureLike | None = None
        self._failures = 0
        self._start = 0.0

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()

    def open(self) -> None:
        capture = self._factory(self.source)
        if not capture.isOpened():
            capture.release()
            raise CameraError(_OPEN_HELP.format(source=self.source))
        self._capture = capture
        self._failures = 0
        self._start = time.monotonic()

    def read(self) -> np.ndarray | None:
        """Return the next BGR frame, or None.

        None means "no frame this time" for a live camera (caller should
        just try again), and "end of video" for a file.
        """
        if self._capture is None:
            raise CameraError("VideoSource.read() called before open()")
        ok, frame = self._capture.read()
        if ok and frame is not None:
            self._failures = 0
            return frame
        if self.is_live:
            self._failures += 1
            if self._failures >= self._max_failures:
                raise CameraError(
                    f"Camera {self.source} stopped delivering frames "
                    f"({self._failures} failed reads in a row). It may have been "
                    "disconnected, or camera permission may have been denied."
                )
        return None

    def timestamp_ms(self) -> int:
        """Timestamp for the most recent frame, in milliseconds.

        Files use their own playback position so results are reproducible;
        live cameras use a monotonic clock.
        """
        if self._capture is not None and not self.is_live:
            return int(self._capture.get(cv2.CAP_PROP_POS_MSEC))
        return int((time.monotonic() - self._start) * 1000)

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
