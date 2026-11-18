"""Pipeline step tests with a fake detector: exercises framing + rendering
without MediaPipe, a model file, or a camera."""

import numpy as np

from faceframe.app import parse_source, process_frame
from faceframe.faces import FaceDetection
from faceframe.framing import FullFrameFramer
from faceframe.geometry import BoundingBox


class FakeDetector:
    def __init__(self, detections):
        self.detections = detections
        self.timestamps = []

    def detect(self, frame, timestamp_ms):
        self.timestamps.append(timestamp_ms)
        return self.detections


def blank():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_process_frame_draws_box_without_mutating_input():
    det = FaceDetection(BoundingBox(100, 100, 80, 80), BoundingBox(100, 100, 80, 80), 0.9, ())
    frame = blank()
    annotated, dets = process_frame(frame, FakeDetector([det]), FullFrameFramer(), 42)
    assert dets == [det]
    assert frame.sum() == 0                          # original untouched
    assert annotated[100, 140].tolist() == [0, 220, 0]  # green box top edge


def test_process_frame_handles_no_detections():
    annotated, dets = process_frame(blank(), FakeDetector([]), FullFrameFramer(), 0)
    assert dets == []
    assert annotated.shape == (480, 640, 3)
    assert (annotated[:, :, 2] > 0).any()  # red "No face detected" text drawn


def test_parse_source():
    assert parse_source("0") == 0
    assert parse_source("2") == 2
    assert parse_source("videos/test.mp4") == "videos/test.mp4"


def test_main_missing_model_exits_2(tmp_path):
    from faceframe.app import main

    assert main(["--model", str(tmp_path / "missing.tflite")]) == 2


def test_main_corrupt_model_exits_2(tmp_path):
    from faceframe.app import main

    bad = tmp_path / "bad.tflite"
    bad.write_bytes(b"not a model")
    assert main(["--model", str(bad)]) == 2


def test_main_missing_video_file_exits_2():
    from faceframe.app import main

    assert main(["--source", "does_not_exist.mp4"]) == 2


def test_min_confidence_out_of_range_is_rejected():
    import pytest

    from faceframe.app import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--min-confidence", "1.5"])
