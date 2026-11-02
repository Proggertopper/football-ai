"""Debug overlay renderer.

Draws state information, lanes, threats, and decision breakdowns
on top of the captured frame using OpenCV.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from config.settings import Config
from decision.actions import ActionSpec
from observe.metrics import PerformanceMetrics
from state.match_state import MatchState
from tactics.modes import TacticalContext
from vision.types import ObjectClass, TrackedObject

logger = logging.getLogger("football_ai.overlay")

try:
    import cv2

    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

# Color constants (BGR)
_GREEN = (0, 255, 0)
_RED = (0, 0, 255)
_BLUE = (255, 0, 0)
_YELLOW = (0, 255, 255)
_WHITE = (255, 255, 255)
_CYAN = (255, 255, 0)
_ORANGE = (0, 165, 255)
_MAGENTA = (255, 0, 255)


class DebugOverlay:
    """Renders debug information onto the frame.

    If OpenCV is not available, all methods are no-ops.
    """

    def __init__(self, cfg: Config) -> None:
        self._enabled: bool = cfg.observe.overlay_enabled and _CV2_AVAILABLE
        self._alpha: float = cfg.observe.overlay_alpha

    def render(
            self,
            frame: np.ndarray,
            state: MatchState,
            tactical: TacticalContext | None = None,
            chosen: ActionSpec | None = None,
            metrics: PerformanceMetrics | None = None,
            tracked: list[TrackedObject] | None = None,
    ) -> np.ndarray:
        """Draw all overlays onto *frame* (modified in place)."""
        if not self._enabled or frame is None:
            return frame

        try:
            # Create a copy for drawing
            overlay = frame.copy()

            if tracked:
                self._draw_tracked_objects(overlay, tracked)
            self._draw_ball(overlay, state)
            self._draw_lanes(overlay, state)
            self._draw_state_panel(overlay, state, tactical, chosen)
            if metrics:
                self._draw_metrics(overlay, metrics)
            if chosen:
                self._draw_action_breakdown(overlay, chosen)

            # Ensure same type and layout before blending
            if overlay.shape != frame.shape:
                overlay = cv2.resize(overlay, (frame.shape[1], frame.shape[0]))
            
            if overlay.dtype != frame.dtype:
                overlay = overlay.astype(frame.dtype)

            # Blend overlay with original frame into a result array first to avoid layout issues
            # then copy back to frame to maintain "in-place" modification
            cv2.addWeighted(overlay, self._alpha, frame, 1 - self._alpha, 0, dst=frame)
            
        except Exception as e:
            logger.error(f"Overlay rendering failed: {e}")
            
        return frame

    def _draw_tracked_objects(self, img: np.ndarray, tracked: list[TrackedObject]) -> None:
        for obj in tracked:
            color = self._object_color(obj.obj_class)
            pos = obj.position.as_int_tuple()
            cv2.circle(img, pos, 8, color, 2)
            # Velocity arrow
            if obj.speed > 0.5:
                end = (obj.position + obj.velocity * 10).as_int_tuple()
                cv2.arrowedLine(img, pos, end, color, 1, tipLength=0.3)
            # Label
            label = f"{obj.obj_class.value[:3]} #{obj.track_id} [{obj.confidence:.2f}]"
            cv2.putText(img, label, (pos[0] + 10, pos[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    @staticmethod
    def _draw_ball(img: np.ndarray, state: MatchState) -> None:
        if state.ball is None:
            return
        pos = state.ball.position.as_int_tuple()
        cv2.circle(img, pos, 12, _YELLOW, 3)
        # Predicted position
        pred = state.predicted_ball_pos.as_int_tuple()
        cv2.circle(img, pred, 6, _ORANGE, 1)
        cv2.line(img, pos, pred, _ORANGE, 1)

    @staticmethod
    def _draw_lanes(img: np.ndarray, state: MatchState) -> None:
        if state.controlled_player is None:
            return
        origin = state.controlled_player.position.as_int_tuple()

        # Passing lanes
        for lane in state.passing_lanes:
            target = lane.target.position.as_int_tuple()
            color = _GREEN if lane.is_clear else _RED
            cv2.line(img, origin, target, color, 1)

        # Shot lanes
        for lane in state.shot_lanes:
            target = lane.target_point.as_int_tuple()
            color = _CYAN if lane.is_clear else _MAGENTA
            cv2.line(img, origin, target, color, 1)

    @staticmethod
    def _draw_state_panel(
            img: np.ndarray,
            state: MatchState,
            tactical: TacticalContext | None,
            chosen: ActionSpec | None,
    ) -> None:
        x, y = 10, 20
        lines = [
            f"Frame: {state.frame_id}",
            f"Zone: {state.controlled_zone.value}",
            f"Possession: {state.possession.value}",
            f"Pressure: {state.local_pressure:.2f}",
            f"Danger: {state.defensive_danger:.2f}",
            f"Counter: {state.counterattack_opportunity:.2f}",
            f"Shot Opp: {state.shot_opportunity:.2f}",
            f"Ball Dist: {state.distance_to_ball:.0f}",
        ]
        if tactical:
            lines.append(f"Mode: {tactical.mode.value}")
        if chosen:
            lines.append(f"Action: {chosen.action_type.value} [{chosen.score:.3f}]")

        for i, line in enumerate(lines):
            cv2.putText(img, line, (x, y + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, _WHITE, 1)

    @staticmethod
    def _draw_metrics(img: np.ndarray, metrics: PerformanceMetrics) -> None:
        h = img.shape[0]
        x, y = 10, h - 80
        lines = [
            f"FPS: {metrics.fps:.1f}",
            f"Inf: {metrics.inference.avg_ms:.1f}ms",
            f"Dec: {metrics.decision.avg_ms:.1f}ms",
            f"Det Conf: {metrics.avg_detection_confidence:.2f}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(img, line, (x, y + i * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, _CYAN, 1)

    @staticmethod
    def _draw_action_breakdown(img: np.ndarray, chosen: ActionSpec) -> None:
        if chosen.metadata is None:
            return
        w = img.shape[1]
        x, y = w - 250, 20
        cv2.putText(img, "Score Breakdown:", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, _YELLOW, 1)
        for i, (k, v) in enumerate(chosen.metadata.items()):
            line = f"  {k}: {v:.3f}"
            cv2.putText(img, line, (x, y + (i + 1) * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, _WHITE, 1)

    @staticmethod
    def _object_color(cls: ObjectClass) -> tuple[int, int, int]:
        match cls:
            case ObjectClass.BALL:
                return _YELLOW
            case ObjectClass.PLAYER_TEAM:
                return _GREEN
            case ObjectClass.PLAYER_OPPONENT:
                return _RED
            case ObjectClass.GOALKEEPER_TEAM:
                return _CYAN
            case ObjectClass.GOALKEEPER_OPPONENT:
                return _ORANGE
            case ObjectClass.REFEREE:
                return _MAGENTA
            case _:
                return _WHITE
