"""Download the official MediaPipe BlazeFace short-range face detection model."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from faceframe.model import DEFAULT_MODEL_PATH, MODEL_URL  # noqa: E402

TFLITE_MAGIC = b"TFL3"  # FlatBuffer file identifier at bytes 4..8 of every .tflite


def main() -> int:
    if DEFAULT_MODEL_PATH.is_file():
        print(f"Model already present: {DEFAULT_MODEL_PATH}")
        return 0
    DEFAULT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = DEFAULT_MODEL_PATH.with_suffix(".part")
    print(f"Downloading {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, tmp)  # errors propagate with a traceback
    header = tmp.read_bytes()[:8]
    if header[4:8] != TFLITE_MAGIC:
        tmp.unlink()
        print("Error: downloaded file is not a TFLite model.", file=sys.stderr)
        return 1
    tmp.rename(DEFAULT_MODEL_PATH)
    print(f"Saved {DEFAULT_MODEL_PATH} ({DEFAULT_MODEL_PATH.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
