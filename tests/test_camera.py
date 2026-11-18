"""VideoSource tests with a fake capture object: no camera hardware needed."""

import numpy as np
import pytest

from faceframe.camera import CameraError, VideoSource

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


class FakeCapture:
    def __init__(self, opened=True, reads=()):
        self._opened = opened
        self._reads = list(reads)
        self.released = False

    def isOpened(self):
        return self._opened

    def read(self):
        return self._reads.pop(0) if self._reads else (False, None)

    def get(self, prop_id):
        return 1234.0

    def release(self):
        self.released = True


def source_with(capture, source=0, **kw):
    return VideoSource(source, capture_factory=lambda _: capture, **kw)


def test_unopenable_camera_raises_with_help_and_releases():
    cap = FakeCapture(opened=False)
    with pytest.raises(CameraError, match="permission"):
        source_with(cap).open()
    assert cap.released


def test_context_manager_releases_on_exit():
    cap = FakeCapture(reads=[(True, FRAME)])
    with source_with(cap) as src:
        assert src.read() is FRAME
    assert cap.released


def test_context_manager_releases_on_exception():
    cap = FakeCapture()
    with pytest.raises(ZeroDivisionError):
        with source_with(cap):
            1 / 0
    assert cap.released


def test_live_camera_tolerates_transient_failures():
    cap = FakeCapture(reads=[(False, None), (False, None), (True, FRAME)])
    with source_with(cap, max_consecutive_failures=3) as src:
        assert src.read() is None
        assert src.read() is None
        assert src.read() is FRAME


def test_live_camera_raises_after_persistent_failures():
    with source_with(FakeCapture(), max_consecutive_failures=3) as src:
        src.read()
        src.read()
        with pytest.raises(CameraError, match="stopped delivering frames"):
            src.read()


def test_video_file_end_returns_none_without_raising():
    with source_with(FakeCapture(), source="clip.mp4", max_consecutive_failures=1) as src:
        assert not src.is_live
        assert src.read() is None
        assert src.read() is None
        assert src.timestamp_ms() == 1234  # uses the file's own position


def test_read_before_open_raises():
    with pytest.raises(CameraError):
        source_with(FakeCapture()).read()
