# FaceFrame: Real-Time Adaptive Camera Framing

An independent prototype exploring automatic camera framing ("keep the
person nicely in frame") for video calls. The project was motivated by a
call where auto-framing became excessively tight, repeatedly zooming toward
a person's nose instead of keeping their whole face in view. FaceFrame is
built from scratch with public tools; it is not based on or a modification
of any vendor's proprietary software.

**Status: Phase 1 (foundation).** Live webcam preview with face detection
and bounding boxes. Cropping, smoothing, and zoom control come in later phases.

## Requirements

- Python 3.10–3.12 recommended (check [mediapipe on PyPI](https://pypi.org/project/mediapipe/) for currently supported versions)
- macOS (Apple Silicon), Linux x86_64/aarch64, or Windows
- A webcam

## Installation

```bash
git clone <your-repo-url> faceframe && cd faceframe
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python scripts/download_model.py   # fetches the official BlazeFace model into models/
```

> **Don't `pip install opencv-python` separately.** MediaPipe already depends
> on `opencv-contrib-python`. Both packages install the same `cv2` module,
> and having both (or the `-headless` variant) is a common cause of
> `cv2.imshow` "function not implemented" errors. If you hit that, run
> `pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python`
> and then `pip install -e ".[dev]"` again.

## Running

```bash
python -m faceframe                 # default webcam
python -m faceframe --mirror        # selfie-style mirrored preview
python -m faceframe --source 1      # a different camera
python -m faceframe --source clip.mp4   # a prerecorded video
```

Press **q** or **Esc**, or close the window, to quit.

**What you should see:** a window titled *FaceFrame* showing your camera.
Each detected face has a green box with a confidence score and six orange
keypoints (eyes, nose tip, mouth, ears/tragions). The top-left status line shows the
face count and per-frame detection time; it turns red and reads
*No face detected* when nobody is in view. A thin blue outline marks the
crop chosen by the framer. In Phase 1 that crop is always the full
frame, so it sits on the window border.

On macOS, the first run triggers a camera permission prompt for your
terminal app (Terminal, iTerm, VS Code...). If you deny it, enable it
under *System Settings → Privacy & Security → Camera* and restart the
terminal.

## Tests

```bash
pytest -v
```

The tests need no camera and no model file: hardware and MediaPipe are
replaced with small fakes at module boundaries.

## Architecture

```
VideoSource ──frame──▶ FaceDetector ──FaceDetection[]──▶ Framer ──crop──▶ render ──▶ cv2.imshow
 camera.py              detector.py      faces.py         framing.py      render.py
                                    (pure geometry in geometry.py)
```

| Module | Responsibility | Touches hardware / ML? |
|---|---|---|
| `geometry.py` | `BoundingBox`, `FrameSize`, clamping, out-of-bounds checks | No |
| `faces.py` | `FaceDetection` type; converts MediaPipe results into it | No (duck-typed) |
| `camera.py` | Opens webcam/video, tolerates dropped frames, releases the device | Yes, camera |
| `detector.py` | MediaPipe Tasks `FaceDetector` in `VIDEO` mode | Yes, MediaPipe |
| `framing.py` | `Framer` interface; Phase 1 `FullFrameFramer` | No |
| `render.py` | Drawing boxes, keypoints, status text | OpenCV drawing only |
| `app.py` | CLI and main loop | Wires everything |

Design choices:

- **MediaPipe Tasks API, `VIDEO` running mode.** Synchronous, processes every
  frame, no callbacks. The same code path runs on live cameras and recorded
  videos, which is what makes reproducible evaluation possible later.
- **Clamped and raw boxes are both kept.** The detector can report boxes that
  extend past the frame edge. `box` is clamped for drawing; `raw_box` is kept
  because how far a face extends off-frame is exactly the "clipping" signal
  the evaluation phase will measure.
- **Privacy.** Frames are processed in memory only; nothing is recorded or
  written to disk.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `Camera error: Could not open video source 0` | No camera, camera in use by another app, or permission denied |
| `Camera ... stopped delivering frames` | Camera unplugged mid-run, or permission revoked |
| `Model error: ... not found` | Run `python scripts/download_model.py` |
| `cv2.imshow` not implemented | Conflicting OpenCV packages; see the install note above |

## Known limitations (Phase 1)

- The short-range BlazeFace model targets faces within about 2 m of the camera;
  people farther away may not be detected.
- No cropping, zoom, smoothing, or identity tracking yet.
- `DEFAULT_MODEL_PATH` assumes an editable install (`pip install -e .`); use
  `--model` otherwise.

## Roadmap

1. **Foundation** (this phase)
2. Baseline framer: crop around the detected faces with fixed margins
3. Improved framer: smoothing, zoom hysteresis, dropout handling, subject selection
4. Evaluation harness on prerecorded clips: latency, crop stability, face clipping
