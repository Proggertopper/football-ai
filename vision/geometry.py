"""2D geometry primitives and utility functions.

All coordinates are in screen-space pixels unless noted otherwise.
Normalized coordinates are in [0..1] range.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class Vec2:
    """Immutable 2D vector / point."""

    x: float
    y: float

    # --- Arithmetic ---

    def __add__(self, other: Vec2) -> Vec2:
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vec2) -> Vec2:
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Vec2:
        return Vec2(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> Vec2:
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> Vec2:
        return Vec2(self.x / scalar, self.y / scalar)

    def __neg__(self) -> Vec2:
        return Vec2(-self.x, -self.y)

    # --- Geometry ---

    def dot(self, other: Vec2) -> float:
        return self.x * other.x + self.y * other.y

    def cross(self, other: Vec2) -> float:
        """2D cross product (scalar z-component)."""
        return self.x * other.y - self.y * other.x

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def length_sq(self) -> float:
        return self.x * self.x + self.y * self.y

    def normalized(self) -> Vec2:
        mag = self.length()
        if mag < 1e-9:
            return Vec2(0.0, 0.0)
        return Vec2(self.x / mag, self.y / mag)

    def angle_deg(self) -> float:
        """Angle in degrees from positive x-axis, range (-180, 180]."""
        return math.degrees(math.atan2(self.y, self.x))

    def rotated(self, angle_deg: float) -> Vec2:
        rad = math.radians(angle_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        return Vec2(self.x * cos_a - self.y * sin_a, self.x * sin_a + self.y * cos_a)

    def lerp(self, other: Vec2, t: float) -> Vec2:
        return Vec2(self.x + (other.x - self.x) * t, self.y + (other.y - self.y) * t)

    def distance_to(self, other: Vec2) -> float:
        return (self - other).length()

    def as_tuple(self) -> tuple[float, float]:
        return self.x, self.y

    def as_int_tuple(self) -> tuple[int, int]:
        return int(round(self.x)), int(round(self.y))

    @staticmethod
    def zero() -> Vec2:
        return Vec2(0.0, 0.0)


@dataclass(frozen=True, slots=True)
class Rect:
    """Axis-aligned bounding box (x, y = top-left corner)."""

    x: float
    y: float
    w: float
    h: float

    @property
    def center(self) -> Vec2:
        return Vec2(self.x + self.w / 2, self.y + self.h / 2)

    @property
    def area(self) -> float:
        return self.w * self.h

    def iou(self, other: Rect) -> float:
        """Intersection over Union."""
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.w, other.x + other.w)
        y2 = min(self.y + self.h, other.y + other.h)
        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter_area = inter_w * inter_h
        union_area = self.area + other.area - inter_area
        if union_area < 1e-9:
            return 0.0
        return inter_area / union_area

    def contains(self, point: Vec2) -> bool:
        return self.x <= point.x <= self.x + self.w and self.y <= point.y <= self.y + self.h


def angle_between(a: Vec2, b: Vec2) -> float:
    """Angle in degrees between two vectors."""
    dot = a.dot(b)
    mag = a.length() * b.length()
    if mag < 1e-9:
        return 0.0
    cos_val = max(-1.0, min(1.0, dot / mag))
    return math.degrees(math.acos(cos_val))


def point_to_segment_distance(point: Vec2, seg_a: Vec2, seg_b: Vec2) -> float:
    """Shortest distance from *point* to the line segment (seg_a, seg_b)."""
    ab = seg_b - seg_a
    ab_len_sq = ab.length_sq()
    if ab_len_sq < 1e-9:
        return point.distance_to(seg_a)
    t = max(0.0, min(1.0, (point - seg_a).dot(ab) / ab_len_sq))
    projection = seg_a + ab * t
    return point.distance_to(projection)


def is_lane_clear(
        origin: Vec2,
        target: Vec2,
        obstacles: Sequence[Vec2],
        lane_width: float,
) -> bool:
    """Check whether a corridor from *origin* to *target* is free of *obstacles*."""
    for obs in obstacles:
        dist = point_to_segment_distance(obs, origin, target)
        if dist < lane_width:
            return False
    return True


def normalize_position(pos: Vec2, screen_w: float, screen_h: float,
                       crop_top: float = 0, crop_bottom: float = 0,
                       crop_left: float = 0, crop_right: float = 0) -> Vec2:
    """Convert screen-space pixel position to [0..1] normalized coordinates."""
    effective_w = screen_w - crop_left - crop_right
    effective_h = screen_h - crop_top - crop_bottom
    if effective_w < 1 or effective_h < 1:
        return Vec2(0.5, 0.5)
    nx = (pos.x - crop_left) / effective_w
    ny = (pos.y - crop_top) / effective_h
    return Vec2(max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny)))