"""Action sequence macros for offline sandbox testing.

Pre-defined input sequences that combine multiple button presses
with precise timing. Used for testing and scenario replay ONLY.
"""

from __future__ import annotations

from dataclasses import dataclass

from control.controller import QueuedInput


@dataclass(frozen=True)
class SequenceStep:
    """One step in a macro sequence."""

    input_type: str  # "press", "release", "left_stick", "right_stick"
    value: str | tuple[float, float]
    delay_frames: int  # frames after sequence start
    hold_frames: int  # how long to hold


class ActionSequence:
    """A named macro sequence of gamepad inputs."""

    def __init__(self, name: str, steps: list[SequenceStep]) -> None:
        self.name = name
        self.steps = steps

    def to_queued_inputs(self, start_frame: int) -> list[QueuedInput]:
        return [
            QueuedInput(
                input_type=step.input_type,
                value=step.value,
                start_frame=start_frame + step.delay_frames,
                duration_frames=step.hold_frames,
            )
            for step in self.steps
        ]


# --- Pre-built sequences for FC 26 ---

DRIVEN_PASS_SEQUENCE = ActionSequence("driven_pass", [
    SequenceStep("press", "RB", delay_frames=0, hold_frames=6),
    SequenceStep("press", "A", delay_frames=0, hold_frames=4),
])

CHIP_THROUGH_BALL_SEQUENCE = ActionSequence("chip_through_ball", [
    SequenceStep("press", "LB", delay_frames=0, hold_frames=6),
    SequenceStep("press", "Y", delay_frames=0, hold_frames=5),
])

POWER_SHOT_SEQUENCE = ActionSequence("power_shot", [
    SequenceStep("press", "LB", delay_frames=0, hold_frames=12),
    SequenceStep("press", "RB", delay_frames=0, hold_frames=12),
    SequenceStep("press", "B", delay_frames=1, hold_frames=10),
])

LOW_DRIVEN = ActionSequence("low_driven_green", [
    SequenceStep("press", "B", delay_frames=0, hold_frames=4),
    # Green timing: second tap at ~200ms = 6 frames at 30fps
    SequenceStep("press", "B", delay_frames=6, hold_frames=2),
])

FAKE_SHOT_SEQUENCE = ActionSequence("fake_shot", [
    SequenceStep("press", "B", delay_frames=0, hold_frames=3),
    SequenceStep("press", "A", delay_frames=2, hold_frames=2),
])

BALL_ROLL_LEFT_SEQUENCE = ActionSequence("ball_roll_left", [
    # Hold RS left relative to player facing direction
    SequenceStep("right_stick", (-1.0, 0.0), delay_frames=0, hold_frames=8),
])

BALL_ROLL_RIGHT_SEQUENCE = ActionSequence("ball_roll_right", [
    SequenceStep("right_stick", (1.0, 0.0), delay_frames=0, hold_frames=8),
])

CONTROLLED_SPRINT_SEQUENCE = ActionSequence("controlled_sprint", [
    SequenceStep("press", "LB", delay_frames=0, hold_frames=15), # Удерживаем L1 для маневров
])

# Registry of all available sequences
SEQUENCE_REGISTRY: dict[str, ActionSequence] = {
    "driven_pass": DRIVEN_PASS_SEQUENCE,
    "chip_through_ball": CHIP_THROUGH_BALL_SEQUENCE,
    "power_shot": POWER_SHOT_SEQUENCE,
    "low_driven_green": LOW_DRIVEN,
    "fake_shot": FAKE_SHOT_SEQUENCE,
    "ball_roll_left": BALL_ROLL_LEFT_SEQUENCE,
    "ball_roll_right": BALL_ROLL_RIGHT_SEQUENCE,
    "controlled_sprint": CONTROLLED_SPRINT_SEQUENCE,
}
