"""FaceFrame entry point: capture -> detect -> frame -> render -> display."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Protocol, Sequence

import cv2
import numpy as np

from faceframe.camera import CameraError, VideoSource
from faceframe.faces import FaceDetection
from faceframe.framing import Framer, FullFrameFramer
from faceframe.geometry import FrameSize
from faceframe.model import DEFAULT_MODEL_PATH
from faceframe.render import draw_crop, draw_detections, draw_status

WINDOW_NAME = "FaceFrame"
QUIT_KEYS = {ord("q"), 27}  # q, Esc


class Detector(Protocol):
    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[FaceDetection]: ...


def process_frame(
    frame: np.ndarray, detector: Detector, framer: Framer, timestamp_ms: int
) -> tuple[np.ndarray, Sequence[FaceDetection]]:
    """One pipeline step. Hardware-free, so it can be tested with a fake detector."""
    size = FrameSize(frame.shape[1], frame.shape[0])
    start = time.perf_counter()
    detections = detector.detect(frame, timestamp_ms)
    detect_ms = (time.perf_counter() - start) * 1000

    crop = framer.compute_crop(detections, size)

    annotated = frame.copy()
    draw_detections(annotated, detections)
    draw_crop(annotated, crop)
    draw_status(annotated, len(detections), detect_ms)
    return annotated, detections


def window_closed() -> bool:
    return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1


def run(source: VideoSource, detector: Detector, framer: Framer, mirror: bool) -> None:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    while True:
        frame = source.read()
        if frame is None:
            if source.is_live:
                continue  # transient drop; VideoSource raises if it persists
            print("End of video.")
            return
        if mirror:
            frame = cv2.flip(frame, 1)  # flip before detection so boxes line up
        annotated, _ = process_frame(frame, detector, framer, source.timestamp_ms())
        cv2.imshow(WINDOW_NAME, annotated)
        if (cv2.waitKey(1) & 0xFF) in QUIT_KEYS or window_closed():
            return


def parse_source(value: str) -> int | str:
    """'0' -> webcam index 0; anything else is treated as a video file path."""
    return int(value) if value.isdigit() else value


def confidence(value: str) -> float:
    score = float(value)
    if not 0.0 <= score <= 1.0:
        raise argparse.ArgumentTypeError(f"must be between 0 and 1, got {score}")
    return score


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="faceframe", description="FaceFrame: real-time face framing prototype")
    p.add_argument("--source", default="0", help="webcam index (default 0) or path to a video file")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="path to the .tflite model")
    p.add_argument("--min-confidence", type=confidence, default=0.5, help="detection score threshold, 0-1")
    p.add_argument("--mirror", action="store_true", help="mirror the preview like a selfie camera")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Imported after arg parsing so `--help` doesn't pay MediaPipe's import cost.
    from faceframe.detector import FaceDetector, ModelLoadError

    source = parse_source(args.source)
    if isinstance(source, str) and not Path(source).is_file():
        print(f"Error: video file not found: {source}", file=sys.stderr)
        return 2
    try:
        with FaceDetector(args.model, args.min_confidence) as detector, VideoSource(source) as video:
            run(video, detector, FullFrameFramer(), args.mirror)
    except (FileNotFoundError, ModelLoadError) as e:
        print(f"Model error: {e}", file=sys.stderr)
        return 2
    except CameraError as e:
        print(f"Camera error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
    return 0
