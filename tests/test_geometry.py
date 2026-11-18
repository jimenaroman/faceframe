import pytest

from faceframe.geometry import (
    BoundingBox,
    FrameSize,
    clamp_to_frame,
    full_frame,
    is_inside,
    normalized_to_pixel,
    visible_fraction,
)

FRAME = FrameSize(640, 480)


def test_box_properties():
    b = BoundingBox(10, 20, 100, 50)
    assert (b.x2, b.y2, b.area, b.center) == (110, 70, 5000, (60.0, 45.0))


@pytest.mark.parametrize("w,h", [(0, 480), (640, 0), (-1, 10)])
def test_frame_size_rejects_non_positive(w, h):
    with pytest.raises(ValueError):
        FrameSize(w, h)


def test_box_fully_inside_is_unchanged():
    b = BoundingBox(100, 100, 200, 150)
    assert is_inside(b, FRAME)
    assert clamp_to_frame(b, FRAME) == b


def test_full_frame_box_is_inside():
    assert is_inside(full_frame(FRAME), FRAME)


def test_negative_origin_is_clamped():
    # face partly off the top-left corner, as MediaPipe can report
    b = BoundingBox(-30, -20, 100, 100)
    assert not is_inside(b, FRAME)
    assert clamp_to_frame(b, FRAME) == BoundingBox(0, 0, 70, 80)


def test_box_past_bottom_right_is_clamped():
    b = BoundingBox(600, 450, 100, 100)
    assert clamp_to_frame(b, FRAME) == BoundingBox(600, 450, 40, 30)


def test_box_larger_than_frame_is_clamped_to_frame():
    assert clamp_to_frame(BoundingBox(-100, -100, 2000, 2000), FRAME) == full_frame(FRAME)


@pytest.mark.parametrize(
    "box",
    [
        BoundingBox(700, 100, 50, 50),   # entirely right of frame
        BoundingBox(-80, 100, 50, 50),   # entirely left
        BoundingBox(100, 480, 50, 50),   # starts exactly at bottom edge (exclusive)
        BoundingBox(100, 100, 0, 50),    # zero width
        BoundingBox(100, 100, 50, -5),   # negative height
    ],
)
def test_invalid_or_outside_boxes_return_none(box):
    assert clamp_to_frame(box, FRAME) is None


def test_visible_fraction():
    assert visible_fraction(BoundingBox(10, 10, 100, 100), FRAME) == 1.0
    assert visible_fraction(BoundingBox(-50, 10, 100, 100), FRAME) == pytest.approx(0.5)
    assert visible_fraction(BoundingBox(1000, 10, 100, 100), FRAME) == 0.0
    assert visible_fraction(BoundingBox(10, 10, 0, 100), FRAME) == 0.0


@pytest.mark.parametrize(
    "x,y,expected",
    [(0.0, 0.0, (0, 0)), (1.0, 1.0, (639, 479)), (0.5, 0.5, (320, 240)),
     (-0.2, 1.5, (0, 479))],  # out-of-range keypoints are clamped
)
def test_normalized_to_pixel(x, y, expected):
    assert normalized_to_pixel(x, y, FRAME) == expected
