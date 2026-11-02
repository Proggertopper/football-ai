"""Scenario definitions for offline evaluation.

Each scenario specifies:
- A fixture (list of TrackedObjects)
- Expected behavior constraints (which actions are acceptable)
- Pass / fail criteria
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from decision.actions import ActionType
from evaluate.fixtures import (
    fixture_blocked_shot,
    fixture_counterattack,
    fixture_crowded_box,
    fixture_loose_ball_race,
    fixture_open_shot,
    fixture_safe_pass_left,
    fixture_under_pressure_defense,
)
from vision.types import TrackedObject


@dataclass
class Scenario:
    """A single test scenario for the evaluation harness."""

    name: str
    description: str
    fixture_fn: Callable[[], list[TrackedObject]]
    # Actions that SHOULD be chosen (at least one must match)
    acceptable_actions: set[ActionType]
    # Actions that must NOT be chosen
    forbidden_actions: set[ActionType] = field(default_factory=set)
    # Minimum score the chosen action should have
    min_score: float = 0.0
    # Maximum risk the chosen action should have
    max_risk: float = 1.0


# ── Scenario Library ─────────────────────────────────────────────

SCENARIOS: list[Scenario] = [
    Scenario(
        name="open_shot",
        description="Clear shooting opportunity inside the box — should shoot",
        fixture_fn=fixture_open_shot,
        acceptable_actions={
            ActionType.SHOOT,
            ActionType.LOW_DRIVEN_SHOT,
            ActionType.FINESSE_SHOT,
            ActionType.TRIVELA_SHOT,
        },
        forbidden_actions={ActionType.CLEAR, ActionType.JOCKEY, ActionType.CONTAIN},
        min_score=0.3,
    ),
    Scenario(
        name="blocked_shot",
        description="Shot partially blocked — should pass, dribble, or attempt a shot",
        fixture_fn=fixture_blocked_shot,
        acceptable_actions={
            ActionType.PASS,
            ActionType.PRECISION_PASS,
            ActionType.THROUGH_BALL,
            ActionType.MOVE,
            ActionType.R1_DRIBBLE,
            # Shooting is acceptable — some lanes may still be partially open
            ActionType.SHOOT,
            ActionType.LOW_DRIVEN_SHOT,
            ActionType.FINESSE_SHOT,
            ActionType.TRIVELA_SHOT,
        },
        forbidden_actions={ActionType.CLEAR, ActionType.JOCKEY, ActionType.CONTAIN},
    ),
    Scenario(
        name="safe_pass_left",
        description="Open teammate on the left — should pass or advance",
        fixture_fn=fixture_safe_pass_left,
        acceptable_actions={
            ActionType.PASS,
            ActionType.PRECISION_PASS,
            ActionType.THROUGH_BALL,
            # With space available, advancing is also valid
            ActionType.SPRINT,
            ActionType.EXPLOSIVE_BURST,
            ActionType.MOVE,
            ActionType.R1_DRIBBLE,
        },
        forbidden_actions={ActionType.CLEAR, ActionType.JOCKEY, ActionType.CONTAIN},
    ),
    Scenario(
        name="under_pressure_defense",
        description="Heavy pressure in own box — should clear or defend",
        fixture_fn=fixture_under_pressure_defense,
        acceptable_actions={
            ActionType.CLEAR,
            ActionType.CONTAIN,
            ActionType.GK_RUSH,
            ActionType.JOCKEY,
            ActionType.SPEED_JOCKEY,
            ActionType.STAND_TACKLE,
            ActionType.SLIDING_TACKLE,
            ActionType.PASS,
            ActionType.PRECISION_PASS,
            ActionType.R1_DRIBBLE,
            ActionType.MOVE,
        },
        forbidden_actions={
            ActionType.SHOOT,
            ActionType.LOW_DRIVEN_SHOT,
            ActionType.THROUGH_BALL,
            ActionType.SPRINT,
        },
    ),
    Scenario(
        name="loose_ball_race",
        description="Loose ball with opponent nearby — should sprint/move toward ball",
        fixture_fn=fixture_loose_ball_race,
        acceptable_actions={
            ActionType.MOVE,
            ActionType.SPRINT,
            ActionType.EXPLOSIVE_BURST,
            ActionType.R1_DRIBBLE,
        },
        forbidden_actions={ActionType.CLEAR, ActionType.SHOOT},
    ),
    Scenario(
        name="crowded_box",
        description="Congested box — should make a quick decision (pass or shoot)",
        fixture_fn=fixture_crowded_box,
        acceptable_actions={
            ActionType.MOVE,
            ActionType.DRIBBLE_TO_GOAL,
            ActionType.SHOOT,
            ActionType.LOW_DRIVEN_SHOT,
            ActionType.PASS,
            ActionType.PRECISION_PASS,
            ActionType.R1_DRIBBLE,
            ActionType.L2_SHIELD,
        },
        forbidden_actions={ActionType.SPRINT, ActionType.CLEAR},
    ),
    Scenario(
        name="counterattack",
        description="Transition with space — should exploit with speed or through ball",
        fixture_fn=fixture_counterattack,
        acceptable_actions={
            ActionType.SPRINT,
            ActionType.EXPLOSIVE_BURST,
            ActionType.THROUGH_BALL,
            ActionType.PASS,
            ActionType.PRECISION_PASS,
            ActionType.MOVE,
            ActionType.R1_DRIBBLE,
        },
        forbidden_actions={ActionType.JOCKEY, ActionType.CONTAIN, ActionType.CLEAR},
    ),
]


def get_scenario(name: str) -> Scenario | None:
    """Look up a scenario by name."""
    for s in SCENARIOS:
        if s.name == name:
            return s
    return None