from faceframe.framing import FullFrameFramer
from faceframe.geometry import BoundingBox, FrameSize


def test_full_frame_framer_never_zooms():
    size = FrameSize(1280, 720)
    assert FullFrameFramer().compute_crop([], size) == BoundingBox(0, 0, 1280, 720)
