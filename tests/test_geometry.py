"""Unit tests for geometry primitives."""

from __future__ import annotations


import pytest

from vision.geometry import (
    Rect,
    Vec2,
    angle_between,
    is_lane_clear,
    normalize_position,
    point_to_segment_distance,
)


class TestVec2:
    def test_add(self) -> None:
        assert Vec2(1, 2) + Vec2(3, 4) == Vec2(4, 6)

    def test_sub(self) -> None:
        assert Vec2(5, 3) - Vec2(2, 1) == Vec2(3, 2)

    def test_mul_scalar(self) -> None:
        assert Vec2(2, 3) * 2 == Vec2(4, 6)

    def test_rmul(self) -> None:
        assert 3 * Vec2(1, 2) == Vec2(3, 6)

    def test_div(self) -> None:
        assert Vec2(6, 4) / 2 == Vec2(3, 2)

    def test_neg(self) -> None:
        assert -Vec2(1, -2) == Vec2(-1, 2)

    def test_length(self) -> None:
        assert Vec2(3, 4).length() == pytest.approx(5.0)

    def test_length_sq(self) -> None:
        assert Vec2(3, 4).length_sq() == pytest.approx(25.0)

    def test_normalized(self) -> None:
        n = Vec2(0, 5).normalized()
        assert n.x == pytest.approx(0.0)
        assert n.y == pytest.approx(1.0)

    def test_normalized_zero(self) -> None:
        n = Vec2(0, 0).normalized()
        assert n == Vec2(0, 0)

    def test_dot(self) -> None:
        assert Vec2(1, 0).dot(Vec2(0, 1)) == pytest.approx(0.0)
        assert Vec2(2, 3).dot(Vec2(4, 5)) == pytest.approx(23.0)

    def test_cross(self) -> None:
        assert Vec2(1, 0).cross(Vec2(0, 1)) == pytest.approx(1.0)

    def test_angle_deg(self) -> None:
        assert Vec2(1, 0).angle_deg() == pytest.approx(0.0)
        assert Vec2(0, 1).angle_deg() == pytest.approx(90.0)
        assert Vec2(-1, 0).angle_deg() == pytest.approx(180.0)

    def test_distance_to(self) -> None:
        assert Vec2(0, 0).distance_to(Vec2(3, 4)) == pytest.approx(5.0)

    def test_lerp(self) -> None:
        result = Vec2(0, 0).lerp(Vec2(10, 10), 0.5)
        assert result == Vec2(5, 5)

    def test_rotated(self) -> None:
        v = Vec2(1, 0).rotated(90)
        assert v.x == pytest.approx(0.0, abs=1e-9)
        assert v.y == pytest.approx(1.0, abs=1e-9)

    def test_as_tuple(self) -> None:
        assert Vec2(1.5, 2.5).as_tuple() == (1.5, 2.5)

    def test_as_int_tuple(self) -> None:
        assert Vec2(1.4, 2.6).as_int_tuple() == (1, 3)


class TestRect:
    def test_center(self) -> None:
        r = Rect(10, 20, 100, 50)
        assert r.center == Vec2(60, 45)

    def test_area(self) -> None:
        assert Rect(0, 0, 10, 20).area == pytest.approx(200.0)

    def test_iou_identical(self) -> None:
        r = Rect(0, 0, 10, 10)
        assert r.iou(r) == pytest.approx(1.0)

    def test_iou_no_overlap(self) -> None:
        r1 = Rect(0, 0, 10, 10)
        r2 = Rect(20, 20, 10, 10)
        assert r1.iou(r2) == pytest.approx(0.0)

    def test_iou_partial(self) -> None:
        r1 = Rect(0, 0, 10, 10)
        r2 = Rect(5, 5, 10, 10)
        # Intersection: 5x5=25, Union: 100+100-25=175
        assert r1.iou(r2) == pytest.approx(25.0 / 175.0)

    def test_contains(self) -> None:
        r = Rect(10, 10, 100, 100)
        assert r.contains(Vec2(50, 50))
        assert not r.contains(Vec2(5, 5))


class TestGeometryFunctions:
    def test_angle_between_perpendicular(self) -> None:
        assert angle_between(Vec2(1, 0), Vec2(0, 1)) == pytest.approx(90.0)

    def test_angle_between_parallel(self) -> None:
        assert angle_between(Vec2(1, 0), Vec2(2, 0)) == pytest.approx(0.0)

    def test_point_to_segment_distance_on_segment(self) -> None:
        dist = point_to_segment_distance(Vec2(5, 5), Vec2(0, 0), Vec2(10, 0))
        assert dist == pytest.approx(5.0)

    def test_point_to_segment_distance_at_endpoint(self) -> None:
        dist = point_to_segment_distance(Vec2(-5, 0), Vec2(0, 0), Vec2(10, 0))
        assert dist == pytest.approx(5.0)

    def test_is_lane_clear_no_obstacles(self) -> None:
        assert is_lane_clear(Vec2(0, 0), Vec2(100, 0), [], 10)

    def test_is_lane_clear_with_obstacle(self) -> None:
        obstacles = [Vec2(50, 5)]  # within lane_width=10
        assert not is_lane_clear(Vec2(0, 0), Vec2(100, 0), obstacles, 10)

    def test_is_lane_clear_obstacle_outside(self) -> None:
        obstacles = [Vec2(50, 50)]  # far from the lane
        assert is_lane_clear(Vec2(0, 0), Vec2(100, 0), obstacles, 10)

    def test_normalize_position(self) -> None:
        result = normalize_position(Vec2(960, 540), 1920, 1080)
        assert result.x == pytest.approx(0.5)
        assert result.y == pytest.approx(0.5)

    def test_normalize_position_with_crop(self) -> None:
        result = normalize_position(
            Vec2(960, 540), 1920, 1080, crop_top=60, crop_bottom=40
        )
        # effective_h = 1080 - 60 - 40 = 980
        # ny = (540 - 60) / 980
        expected_y = 480 / 980
        assert result.y == pytest.approx(expected_y, abs=0.01)