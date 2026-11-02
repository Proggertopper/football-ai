"""Controller abstraction layer.

Translates ActionSpecs from the decision engine into concrete gamepad
inputs (button presses, stick values, timed sequences).

FIX NOTES:
- execute() NO LONGER calls gamepad.reset() + update() which was sending
  a neutral frame (all zeros) to the game before every action, killing
  all movement.
- Continuous actions (MOVE, SPRINT, JOCKEY, etc.) keep the stick held
  without dead frames between executions.
- The stick direction is applied IMMEDIATELY in execute() so the very
  first tick already has the stick set.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import Protocol

from config.settings import Config
from decision.actions import ActionSpec, ActionType

_log = logging.getLogger("football_ai.controller")


class GamepadProtocol(Protocol):
    """Minimal interface any gamepad backend must implement."""

    def press_button(self, button: str) -> None: ...

    def release_button(self, button: str) -> None: ...

    def set_left_stick(self, x: float, y: float) -> None: ...

    def set_right_stick(self, x: float, y: float) -> None: ...

    def reset(self) -> None: ...

    def update(self) -> None: ...


@dataclass
class QueuedInput:
    """A single gamepad input with timing."""

    input_type: str  # "press", "release", "left_stick", "right_stick"
    value: str | tuple[float, float]
    start_frame: int
    duration_frames: int
    executed: bool = False


class ActionController:
    """Executes actions by translating them to gamepad inputs.

    Maintains an input queue and processes it each frame.
    Only one action sequence runs at a time; new actions interrupt
    if the current action is interruptible.
    """

    def __init__(self, cfg: Config, gamepad: GamepadProtocol) -> None:
        self._gamepad = gamepad
        self._cfg = cfg
        self._queue: list[QueuedInput] = []
        self._current_action: ActionType | None = None
        self._current_interruptible: bool = True
        self._frame: int = 0
        self._action_end_frame: int = 0
        self._freeze_left: bool = False
        self._ls_smooth: deque[tuple[float, float]] = deque(maxlen=3)

    def _emit_left_stick(self, x: float, y: float) -> None:
        """3-frame moving average on left stick for human-like smoothing."""
        self._ls_smooth.append((max(-1.0, min(1.0, x)), max(-1.0, min(1.0, y))))
        ax = sum(t[0] for t in self._ls_smooth) / len(self._ls_smooth)
        ay = sum(t[1] for t in self._ls_smooth) / len(self._ls_smooth)
        _log.info(
            "left_stick frame=%s action=%s x=%.4f y=%.4f",
            self._frame,
            self._current_action.value if self._current_action else None,
            ax,
            ay,
        )
        self._gamepad.set_left_stick(ax, ay)

    def execute(self, spec: ActionSpec, freeze_left_stick: bool = False) -> bool:
        """Begin executing an action. Returns False if blocked."""
        if not self._current_interruptible and self._frame < self._action_end_frame:
            return False

        # Release buttons from the OLD action, but do NOT zero the stick
        # yet — _build_inputs will set the new stick value immediately.
        # This eliminates the "neutral frame" that was killing all movement.
        self._release_old_buttons()
        self._queue.clear()

        self._current_action = spec.action_type
        self._current_interruptible = spec.interruptible
        self._action_end_frame = self._frame + spec.duration_frames
        self._freeze_left = bool(freeze_left_stick)

        self._build_inputs(spec)

        # Apply stick direction IMMEDIATELY so the first tick() already
        # has the correct stick position (no 1-frame delay).
        if spec.direction is not None:
            if spec.action_type == ActionType.RS_SWITCH:
                self._gamepad.set_right_stick(spec.direction[0], spec.direction[1])
            elif not self._freeze_left:
                self._emit_left_stick(spec.direction[0], spec.direction[1])
            else:
                self._ls_smooth.clear()
                self._emit_left_stick(0.0, 0.0)

        return True

    def tick(self) -> None:
        """Process one frame of the input queue."""
        self._frame += 1

        active = False
        for inp in self._queue:
            if inp.executed:
                continue
            if self._frame >= inp.start_frame:
                if self._frame < inp.start_frame + inp.duration_frames:
                    self._apply_input(inp)
                    active = True
                else:
                    self._release_input(inp)
                    inp.executed = True

        # Cleanup completed inputs
        self._queue = [q for q in self._queue if not q.executed]

        if not active and self._frame >= self._action_end_frame:
            self._current_action = None

        if self._freeze_left:
            self._ls_smooth.clear()
            _log.info(
                "left_stick frame=%s FREEZE x=0.0000 y=0.0000",
                self._frame,
            )
            self._gamepad.set_left_stick(0.0, 0.0)

        # Commit all changes to the virtual gamepad
        self._gamepad.update()

    def cancel(self) -> None:
        """Immediately cancel all inputs and reset to neutral."""
        self._queue.clear()
        self._ls_smooth.clear()
        self._freeze_left = False
        self._gamepad.reset()
        self._gamepad.update()
        self._current_action = None
        self._current_interruptible = True

    @property
    def is_busy(self) -> bool:
        return self._current_action is not None and self._frame < self._action_end_frame

    # --- Private helpers ---

    def _release_old_buttons(self) -> None:
        """Release buttons from the previous action without touching the stick."""
        for inp in self._queue:
            if inp.executed:
                continue
            if inp.input_type == "press" and isinstance(inp.value, str):
                self._gamepad.release_button(inp.value)

    def _build_inputs(self, spec: ActionSpec) -> None:
        """Translate an ActionSpec into queued gamepad inputs."""
        f = self._frame

        if spec.direction is not None and spec.action_type != ActionType.RS_SWITCH:
            self._queue.append(QueuedInput(
                input_type="left_stick",
                value=spec.direction,
                start_frame=f,
                duration_frames=spec.duration_frames,
            ))

        match spec.action_type:
            case ActionType.MOVE:
                pass  # stick direction only

            case ActionType.DRIBBLE_TO_GOAL:
                self._queue.append(QueuedInput("press", "RB", f, max(4, spec.duration_frames - 2)))
                # left stick queued above

            case ActionType.SPRINT:
                self._queue.append(QueuedInput("press", "RT", f, spec.duration_frames))

            case ActionType.R1_DRIBBLE:
                self._queue.append(QueuedInput("press", "RB", f, spec.duration_frames))

            case ActionType.EXPLOSIVE_BURST:
                # R1 tap then R2
                self._queue.append(QueuedInput("press", "RB", f, 2))
                self._queue.append(QueuedInput("press", "RT", f + 2, max(1, spec.duration_frames - 2)))

            case ActionType.JOCKEY:
                self._queue.append(QueuedInput("press", "LT", f, spec.duration_frames))

            case ActionType.SPEED_JOCKEY:
                self._queue.append(QueuedInput("press", "LT", f, spec.duration_frames))
                self._queue.append(QueuedInput("press", "RT", f, spec.duration_frames))

            case ActionType.PASS:
                self._queue.append(QueuedInput("press", "A", f, spec.duration_frames))

            case ActionType.PRECISION_PASS:
                self._queue.append(QueuedInput("press", "RB", f, spec.duration_frames))
                self._queue.append(QueuedInput("press", "Y", f, spec.duration_frames))

            case ActionType.THROUGH_BALL:
                self._queue.append(QueuedInput("press", "Y", f, spec.duration_frames))

            case ActionType.SHOOT:
                self._queue.append(QueuedInput("press", "B", f, spec.duration_frames))

            case ActionType.LOW_DRIVEN_SHOT:
                # Double-tap B with green timing
                self._queue.append(QueuedInput("press", "B", f, 3))
                self._queue.append(QueuedInput("press", "B", f + 6, 2))

            case ActionType.FINESSE_SHOT:
                self._queue.append(QueuedInput("press", "RB", f, spec.duration_frames))
                self._queue.append(QueuedInput("press", "B", f, spec.duration_frames))

            case ActionType.TRIVELA_SHOT:
                self._queue.append(QueuedInput("press", "LT", f, spec.duration_frames))
                self._queue.append(QueuedInput("press", "B", f, spec.duration_frames))

            case ActionType.STAND_TACKLE:
                self._queue.append(QueuedInput("press", "B", f, spec.duration_frames))

            case ActionType.SLIDING_TACKLE:
                self._queue.append(QueuedInput("press", "B", f, spec.duration_frames))

            case ActionType.CLEAR:
                self._queue.append(QueuedInput("press", "B", f, spec.duration_frames))

            case ActionType.CONTAIN:
                self._queue.append(QueuedInput("press", "A", f, spec.duration_frames))

            case ActionType.SWITCH_PLAYER:
                self._queue.append(QueuedInput("press", "LB", f, 2))

            case ActionType.RS_SWITCH:
                # Pro switch: full-magnitude RS flick (right_joystick) then return to neutral.
                if spec.direction is not None:
                    fx, fy = float(spec.direction[0]), float(spec.direction[1])
                    mag = math.hypot(fx, fy)
                    if mag > 1e-6:
                        fx, fy = fx / mag, fy / mag
                    total = max(7, spec.duration_frames)
                    flick = max(4, min(6, total - 3))
                    settle = max(2, total - flick)
                    self._queue.append(
                        QueuedInput("right_stick", (fx, fy), f, flick)
                    )
                    self._queue.append(
                        QueuedInput("right_stick", (0.0, 0.0), f + flick, settle)
                    )

            case ActionType.L2_SHIELD:
                self._queue.append(QueuedInput("press", "LT", f, spec.duration_frames))

            case ActionType.CANCEL | ActionType.SUPER_CANCEL:
                self._queue.append(QueuedInput("press", "LT", f, 2))
                self._queue.append(QueuedInput("press", "RT", f, 2))

            case ActionType.GK_NUDGE:
                self._queue.append(QueuedInput("press", "RS_CLICK", f, 2))

    def _apply_input(self, inp: QueuedInput) -> None:
        match inp.input_type:
            case "press":
                assert isinstance(inp.value, str)
                self._gamepad.press_button(inp.value)
            case "left_stick":
                assert isinstance(inp.value, tuple)
                if not self._freeze_left:
                    self._emit_left_stick(inp.value[0], inp.value[1])
                else:
                    self._ls_smooth.clear()
                    self._emit_left_stick(0.0, 0.0)
            case "right_stick":
                assert isinstance(inp.value, tuple)
                rx, ry = float(inp.value[0]), float(inp.value[1])
                _log.info("right_stick frame=%s x=%.4f y=%.4f", self._frame, rx, ry)
                self._gamepad.set_right_stick(rx, ry)

    def _release_input(self, inp: QueuedInput) -> None:
        match inp.input_type:
            case "press":
                assert isinstance(inp.value, str)
                self._gamepad.release_button(inp.value)
