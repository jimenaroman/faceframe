"""Tests for the MediaPipe -> FaceDetection adapter, using fake objects shaped
like mediapipe.tasks Detection results (no model or camera needed)."""

from types import SimpleNamespace as NS

from faceframe.faces import convert_detections
from faceframe.geometry import BoundingBox, FrameSize

FRAME = FrameSize(640, 480)


def fake_detection(x, y, w, h, score=0.9, keypoints=((0.5, 0.5),)):
    return NS(
        bounding_box=NS(origin_x=x, origin_y=y, width=w, height=h),
        categories=[NS(score=score)] if score is not None else [],
        keypoints=[NS(x=kx, y=ky) for kx, ky in keypoints],
    )


def test_converts_in_frame_detection():
    [d] = convert_detections([fake_detection(100, 80, 120, 140)], FRAME, 0.5)
    assert d.box == d.raw_box == BoundingBox(100, 80, 120, 140)
    assert d.score == 0.9
    assert d.keypoints == ((0.5, 0.5),)


def test_partially_out_of_bounds_is_clamped_but_raw_kept():
    [d] = convert_detections([fake_detection(-40, 400, 120, 140)], FRAME, 0.5)
    assert d.raw_box == BoundingBox(-40, 400, 120, 140)
    assert d.box == BoundingBox(0, 400, 80, 80)


def test_fully_out_of_bounds_is_dropped():
    assert convert_detections([fake_detection(900, 900, 50, 50)], FRAME, 0.5) == []


def test_low_score_and_missing_category_are_dropped():
    dets = [fake_detection(10, 10, 50, 50, score=0.3), fake_detection(10, 10, 50, 50, score=None)]
    assert convert_detections(dets, FRAME, 0.5) == []


def test_empty_result_gives_empty_list():
    assert convert_detections([], FRAME, 0.5) == []


def test_keypoints_with_missing_coords_are_skipped():
    det = fake_detection(10, 10, 50, 50, keypoints=((0.1, 0.2),))
    det.keypoints.append(NS(x=None, y=0.3))
    [d] = convert_detections([det], FRAME, 0.5)
    assert d.keypoints == ((0.1, 0.2),)


def test_multiple_faces_preserve_order():
    dets = [fake_detection(10, 10, 50, 50), fake_detection(300, 100, 60, 60)]
    out = convert_detections(dets, FRAME, 0.5)
    assert [d.box.x for d in out] == [10, 300]
