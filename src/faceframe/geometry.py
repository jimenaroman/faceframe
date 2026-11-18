"""Pure geometry helpers. No OpenCV, no MediaPipe, no hardware.

Everything here is deterministic and unit-tested. Coordinates are integer
pixels with the origin at the top-left; `x2`/`y2` are exclusive, so a box
covering a whole 640x480 frame is BoundingBox(0, 0, 640, 480).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrameSize:
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Frame size must be positive, got {self.width}x{self.height}")


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return max(self.width, 0) * max(self.height, 0)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

    def has_positive_area(self) -> bool:
        return self.width > 0 and self.height > 0


def full_frame(size: FrameSize) -> BoundingBox:
    return BoundingBox(0, 0, size.width, size.height)


def is_inside(box: BoundingBox, size: FrameSize) -> bool:
    """True if the box lies completely within the frame."""
    return box.x >= 0 and box.y >= 0 and box.x2 <= size.width and box.y2 <= size.height


def clamp_to_frame(box: BoundingBox, size: FrameSize) -> BoundingBox | None:
    """Intersect a box with the frame.

    Returns None when the box has no area or lies entirely outside the
    frame. Detectors legitimately return boxes that extend past the edge
    (e.g. a face half out of shot), so this is expected, not an error.
    """
    if not box.has_positive_area():
        return None
    x1 = max(box.x, 0)
    y1 = max(box.y, 0)
    x2 = min(box.x2, size.width)
    y2 = min(box.y2, size.height)
    if x2 <= x1 or y2 <= y1:
        return None
    return BoundingBox(x1, y1, x2 - x1, y2 - y1)


def visible_fraction(box: BoundingBox, size: FrameSize) -> float:
    """Fraction (0..1) of the box's area that is inside the frame.

    1.0 means fully visible; values below 1.0 mean the face is clipped by
    the frame edge. This is the seed of the later "face clipping" metric.
    """
    if not box.has_positive_area():
        return 0.0
    clamped = clamp_to_frame(box, size)
    if clamped is None:
        return 0.0
    return clamped.area / box.area


def normalized_to_pixel(x: float, y: float, size: FrameSize) -> tuple[int, int]:
    """Convert a normalized [0, 1] point to pixel coordinates, clamped to the frame."""
    px = int(round(x * (size.width - 1)))
    py = int(round(y * (size.height - 1)))
    return (min(max(px, 0), size.width - 1), min(max(py, 0), size.height - 1))
