"""Synthetic detection fixtures for offline testing.

Provides factory functions that build realistic TrackedObject lists
for specific scenarios (open shot, crowded box, etc.) so the decision
engine and state estimator can be tested deterministically without
a live game or YOLO model.

All layout coordinates are authored in a *virtual* 1920×1080 pitch, then
mapped to ``config/default.yaml`` world size (1280×720) so they match
``StateEstimator`` zone and shot-lane math.
"""

from __future__ import annotations

from typing import Callable, Dict

from vision.geometry import Rect, Vec2
from vision.types import ObjectClass, TrackedObject

_FXS = 1280.0 / 1920.0
_FYS = 720.0 / 1080.0


def _fx(x: float) -> float:
    return x * _FXS


def _fy(y: float) -> float:
    return y * _FYS


def _make_player(
        track_id: int,
        cls: ObjectClass,
        x: float,
        y: float,
        vx: float = 0.0,
        vy: float = 0.0,
        confidence: float = 0.92,
) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        obj_class=cls,
        position=Vec2(x, y),
        velocity=Vec2(vx, vy),
        bbox=Rect(x - 15, y - 30, 30, 60),
        confidence=confidence,
        age=20,
        hits=18,
        time_since_update=0,
        predicted=False,
    )


def _make_ball(
        x: float, y: float, vx: float = 0.0, vy: float = 0.0, confidence: float = 0.95,
) -> TrackedObject:
    return TrackedObject(
        track_id=100,
        obj_class=ObjectClass.BALL,
        position=Vec2(x, y),
        velocity=Vec2(vx, vy),
        bbox=Rect(x - 5, y - 5, 10, 10),
        confidence=confidence,
        age=30,
        hits=28,
        time_since_update=0,
        predicted=False,
    )


# ── Scenario Fixtures ───────────────────────────────────────────────


def fixture_open_shot() -> list[TrackedObject]:
    """Controlled player in the box with a clear shot on goal (mapped to 1280×720)."""
    return [
        _make_ball(_fx(1700), _fy(540), vx=_fx(1.0), vy=_fy(0.0)),
        _make_player(1, ObjectClass.PLAYER_TEAM, _fx(1690), _fy(540), vx=_fx(2.0), vy=_fy(0.0)),
        _make_player(2, ObjectClass.PLAYER_TEAM, _fx(1500), _fy(400)),
        _make_player(3, ObjectClass.PLAYER_TEAM, _fx(1500), _fy(680)),
        _make_player(10, ObjectClass.PLAYER_OPPONENT, _fx(1300), _fy(500)),
        _make_player(11, ObjectClass.PLAYER_OPPONENT, _fx(1300), _fy(600)),
        _make_player(12, ObjectClass.PLAYER_OPPONENT, _fx(1100), _fy(540)),
        _make_player(20, ObjectClass.GOALKEEPER_OPPONENT, _fx(1890), _fy(540)),
    ]


def fixture_blocked_shot() -> list[TrackedObject]:
    """Controlled player in shooting position but lane is blocked."""
    return [
        _make_ball(_fx(1650), _fy(500), vx=_fx(0.5), vy=_fy(0.0)),
        _make_player(1, ObjectClass.PLAYER_TEAM, _fx(1640), _fy(500), vx=_fx(1.5), vy=_fy(0.0)),
        _make_player(2, ObjectClass.PLAYER_TEAM, _fx(1400), _fy(350)),
        _make_player(3, ObjectClass.PLAYER_TEAM, _fx(1400), _fy(650)),
        _make_player(10, ObjectClass.PLAYER_OPPONENT, _fx(1750), _fy(510)),
        _make_player(11, ObjectClass.PLAYER_OPPONENT, _fx(1800), _fy(520)),
        _make_player(12, ObjectClass.PLAYER_OPPONENT, _fx(1600), _fy(400)),
        _make_player(20, ObjectClass.GOALKEEPER_OPPONENT, _fx(1890), _fy(540)),
    ]


def fixture_safe_pass_left() -> list[TrackedObject]:
    """Clear passing lane to a teammate on the left wing (not center-circle)."""
    return [
        _make_ball(_fx(1300), _fy(520)),
        _make_player(1, ObjectClass.PLAYER_TEAM, _fx(1280), _fy(520)),
        _make_player(2, ObjectClass.PLAYER_TEAM, _fx(800), _fy(380)),
        _make_player(3, ObjectClass.PLAYER_TEAM, _fx(1100), _fy(540)),
        _make_player(4, ObjectClass.PLAYER_TEAM, _fx(700), _fy(700)),
        _make_player(10, ObjectClass.PLAYER_OPPONENT, _fx(1200), _fy(500)),
        _make_player(11, ObjectClass.PLAYER_OPPONENT, _fx(1200), _fy(600)),
        _make_player(12, ObjectClass.PLAYER_OPPONENT, _fx(1000), _fy(300)),
        _make_player(20, ObjectClass.GOALKEEPER_OPPONENT, _fx(1890), _fy(540)),
    ]


def fixture_under_pressure_defense() -> list[TrackedObject]:
    """Controlled player in own box under heavy pressure."""
    return [
        _make_ball(_fx(250), _fy(520), vx=_fx(-1.0), vy=_fy(0.0)),
        _make_player(1, ObjectClass.PLAYER_TEAM, _fx(240), _fy(520)),
        _make_player(2, ObjectClass.PLAYER_TEAM, _fx(150), _fy(400)),
        _make_player(3, ObjectClass.PLAYER_TEAM, _fx(150), _fy(640)),
        _make_player(10, ObjectClass.PLAYER_OPPONENT, _fx(300), _fy(500), vx=_fx(-3.0), vy=_fy(0.0)),
        _make_player(11, ObjectClass.PLAYER_OPPONENT, _fx(320), _fy(560), vx=_fx(-2.5), vy=_fy(0.0)),
        _make_player(12, ObjectClass.PLAYER_OPPONENT, _fx(350), _fy(480), vx=_fx(-2.0), vy=_fy(0.0)),
        _make_player(13, ObjectClass.PLAYER_OPPONENT, _fx(400), _fy(540)),
        _make_player(19, ObjectClass.GOALKEEPER_TEAM, _fx(50), _fy(540)),
        _make_player(20, ObjectClass.GOALKEEPER_OPPONENT, _fx(1890), _fy(540)),
    ]


def fixture_loose_ball_race() -> list[TrackedObject]:
    """Ball is loose in midfield — both teams racing to it."""
    return [
        _make_ball(_fx(900), _fy(500), vx=_fx(3.0), vy=_fy(-0.5)),
        _make_player(1, ObjectClass.PLAYER_TEAM, _fx(850), _fy(510), vx=_fx(4.0), vy=_fy(-0.5)),
        _make_player(2, ObjectClass.PLAYER_TEAM, _fx(700), _fy(400)),
        _make_player(3, ObjectClass.PLAYER_TEAM, _fx(600), _fy(600)),
        _make_player(10, ObjectClass.PLAYER_OPPONENT, _fx(860), _fy(490), vx=_fx(3.5), vy=_fy(0.5)),
        _make_player(11, ObjectClass.PLAYER_OPPONENT, _fx(1000), _fy(500)),
        _make_player(12, ObjectClass.PLAYER_OPPONENT, _fx(1100), _fy(550)),
        _make_player(20, ObjectClass.GOALKEEPER_OPPONENT, _fx(1890), _fy(540)),
    ]


def fixture_crowded_box() -> list[TrackedObject]:
    """Attacking in the opponent's box — very congested."""
    return [
        _make_ball(_fx(1750), _fy(520), vx=_fx(0.5), vy=_fy(0.0)),
        _make_player(1, ObjectClass.PLAYER_TEAM, _fx(1740), _fy(520)),
        _make_player(2, ObjectClass.PLAYER_TEAM, _fx(1720), _fy(460)),
        _make_player(3, ObjectClass.PLAYER_TEAM, _fx(1710), _fy(600)),
        _make_player(4, ObjectClass.PLAYER_TEAM, _fx(1680), _fy(540)),
        _make_player(10, ObjectClass.PLAYER_OPPONENT, _fx(1760), _fy(500)),
        _make_player(11, ObjectClass.PLAYER_OPPONENT, _fx(1770), _fy(560)),
        _make_player(12, ObjectClass.PLAYER_OPPONENT, _fx(1730), _fy(480)),
        _make_player(13, ObjectClass.PLAYER_OPPONENT, _fx(1750), _fy(580)),
        _make_player(14, ObjectClass.PLAYER_OPPONENT, _fx(1800), _fy(530)),
        _make_player(20, ObjectClass.GOALKEEPER_OPPONENT, _fx(1890), _fy(540)),
    ]


def fixture_counterattack() -> list[TrackedObject]:
    """Transition moment — just won the ball, space to exploit."""
    return [
        _make_ball(_fx(600), _fy(540), vx=_fx(2.0), vy=_fy(0.0)),
        _make_player(1, ObjectClass.PLAYER_TEAM, _fx(590), _fy(540), vx=_fx(4.0), vy=_fy(0.0)),
        _make_player(2, ObjectClass.PLAYER_TEAM, _fx(500), _fy(350), vx=_fx(3.0), vy=_fy(0.0)),
        _make_player(3, ObjectClass.PLAYER_TEAM, _fx(500), _fy(700), vx=_fx(3.0), vy=_fy(0.0)),
        _make_player(10, ObjectClass.PLAYER_OPPONENT, _fx(1400), _fy(500)),
        _make_player(11, ObjectClass.PLAYER_OPPONENT, _fx(1500), _fy(600)),
        _make_player(12, ObjectClass.PLAYER_OPPONENT, _fx(700), _fy(400)),
        _make_player(20, ObjectClass.GOALKEEPER_OPPONENT, _fx(1890), _fy(540)),
    ]


def fixture_set_piece_corner() -> list[TrackedObject]:
    """Corner kick setup — static positions."""
    return [
        _make_ball(_fx(1900), _fy(50)),
        _make_player(1, ObjectClass.PLAYER_TEAM, _fx(1895), _fy(55)),
        _make_player(2, ObjectClass.PLAYER_TEAM, _fx(1800), _fy(500)),
        _make_player(3, ObjectClass.PLAYER_TEAM, _fx(1780), _fy(560)),
        _make_player(4, ObjectClass.PLAYER_TEAM, _fx(1810), _fy(480)),
        _make_player(10, ObjectClass.PLAYER_OPPONENT, _fx(1820), _fy(510)),
        _make_player(11, ObjectClass.PLAYER_OPPONENT, _fx(1790), _fy(550)),
        _make_player(12, ObjectClass.PLAYER_OPPONENT, _fx(1830), _fy(530)),
        _make_player(20, ObjectClass.GOALKEEPER_OPPONENT, _fx(1870), _fy(540)),
    ]


def fixture_gk_emergency() -> list[TrackedObject]:
    """Opponent break — GK rush heuristic smoke test."""
    return [
        _make_ball(_fx(1850), _fy(540), vx=_fx(5.0), vy=_fy(0.0)),
        _make_player(10, ObjectClass.PLAYER_OPPONENT, _fx(1800), _fy(540)),
        _make_player(20, ObjectClass.GOALKEEPER_TEAM, _fx(1890), _fy(540)),
    ]


FIXTURE_REGISTRY: dict[str, type] = {}

ALL_FIXTURES: Dict[str, Callable[[], list[TrackedObject]]] = {
    "open_shot": fixture_open_shot,
    "blocked_shot": fixture_blocked_shot,
    "safe_pass_left": fixture_safe_pass_left,
    "under_pressure_defense": fixture_under_pressure_defense,
    "loose_ball_race": fixture_loose_ball_race,
    "crowded_box": fixture_crowded_box,
    "counterattack": fixture_counterattack,
    "set_piece_corner": fixture_set_piece_corner,
}
