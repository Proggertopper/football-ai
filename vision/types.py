"""Core data types for the vision pipeline.

Detection  — raw output from YOLO (per-frame, no identity).
TrackedObject — persistent identity across frames with velocity.
ObjectClass — semantic label enum.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from vision.geometry import Rect, Vec2


class ObjectClass(enum.Enum):
    """Semantic object classes recognized by the detector."""

    BALL = "ball"
    PLAYER_TEAM = "player_team"
    PLAYER_OPPONENT = "player_opponent"
    GOALKEEPER_TEAM = "goalkeeper_team"
    GOALKEEPER_OPPONENT = "goalkeeper_opponent"
    REFEREE = "referee"
    UNKNOWN = "unknown"

    # @staticmethod
    # def from_class_id(class_id: int, class_map: dict[int, str]) -> ObjectClass:
    #     label = class_map.get(class_id, "unknown")
    #     try:
    #         return ObjectClass(label)
    #     except ValueError:
    #         return ObjectClass.UNKNOWN
    #
    # def is_teammate(self) -> bool:
    #     return self in (ObjectClass.PLAYER_TEAM, ObjectClass.GOALKEEPER_TEAM)
    # 
    # def is_opponent(self) -> bool:
    #     return self in (ObjectClass.PLAYER_OPPONENT, ObjectClass.GOALKEEPER_OPPONENT)

    @staticmethod
    def from_class_id(class_id: int, class_map: dict[int, str]) -> ObjectClass:
        label = class_map.get(class_id, "unknown")
        try:
            # ПРОВЕРКА: печатает, что мы пытаемся превратить в Enum
            # print(f"DEBUG: Mapping ID {class_id} -> label '{label}'")
            return ObjectClass(label)
        except ValueError:
            return ObjectClass.UNKNOWN


@dataclass(slots=True)
class Detection:
    """Single raw detection from the YOLO model in one frame."""

    bbox: Rect
    class_id: int
    confidence: float
    frame_id: int

    @property
    def center(self) -> Vec2:
        return self.bbox.center


@dataclass(slots=True)
class TrackedObject:
    """An object with persistent identity across frames.

    Updated each frame by the tracker with smoothed position,
    estimated velocity, and age statistics.
    """

    track_id: int
    obj_class: ObjectClass
    position: Vec2  # smoothed screen-space center
    velocity: Vec2 = field(default_factory=Vec2.zero)  # pixels/frame
    bbox: Rect = field(default_factory=lambda: Rect(0, 0, 0, 0))
    confidence: float = 0.0
    age: int = 0  # total frames this track has lived
    hits: int = 0  # frames with a matched detection
    time_since_update: int = 0  # frames since last detection match
    predicted: bool = False  # True if position is extrapolated

    @property
    def speed(self) -> float:
        return self.velocity.length()

    @property
    def direction(self) -> Vec2:
        return self.velocity.normalized()

    def predict_position(self, frames_ahead: int = 1) -> Vec2:
        """Linear extrapolation of position."""
        return self.position + self.velocity * frames_ahead