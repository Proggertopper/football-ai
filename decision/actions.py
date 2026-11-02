"""Action definitions — every legal action the AI can take.

Each action has:
- preconditions (when it's legal)
- cooldown (frames before re-use)
- duration (how long to hold the input)
- interruptibility
- risk score function
- expected value function
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from state.match_state import MatchState, PossessionState


class ActionType(enum.Enum):
    """All legal actions for the offline sandbox."""

    MOVE = "move"
    SPRINT = "sprint"
    JOCKEY = "jockey"
    STAND_TACKLE = "stand_tackle"
    SLIDING_TACKLE = "sliding_tackle"
    PASS = "pass"
    THROUGH_BALL = "through_ball"
    SHOOT = "shoot"
    CLEAR = "clear"
    CONTAIN = "contain"
    SWITCH_PLAYER = "switch_player"
    CANCEL = "cancel"
    # FC26-specific compound actions
    PRECISION_PASS = "precision_pass"
    LOW_DRIVEN_SHOT = "low_driven_shot"
    FINESSE_SHOT = "finesse_shot"
    TRIVELA_SHOT = "trivela_shot"
    R1_DRIBBLE = "r1_dribble"
    EXPLOSIVE_BURST = "explosive_burst"
    L2_SHIELD = "l2_shield"
    SPEED_JOCKEY = "speed_jockey"
    SUPER_CANCEL = "super_cancel"
    RS_SWITCH = "rs_switch"
    GK_NUDGE = "gk_nudge"
    GK_RUSH = "gk_rush"
    DRIBBLE_TO_GOAL = "dribble_to_goal"


@dataclass(frozen=True, slots=True)
class ActionSpec:
    """Specification for a scored action candidate.

    Built by the scorer, consumed by the decision engine.
    """

    action_type: ActionType
    score: float  # overall desirability (higher = better)
    risk: float  # 0–1 risk level
    expected_value: float  # estimated reward
    direction: tuple[float, float] | None = None  # stick direction if needed
    target_position: tuple[float, float] | None = None  # pass/shot target
    duration_frames: int = 4
    interruptible: bool = True
    metadata: dict[str, float] | None = None  # breakdown for debugging


# --- Precondition functions ---
# Each returns True if the action is legal in the current state.

def can_move(state: MatchState) -> bool:
    return state.controlled_player is not None


def can_sprint(state: MatchState) -> bool:
    return state.controlled_player is not None


def can_jockey(state: MatchState) -> bool:
    return state.controlled_player is not None


def can_tackle(state: MatchState, confidence_min: float = 0.85) -> bool:
    """Stand or slide: only when very close to the opponent nearest the ball."""
    if state.controlled_player is None or state.ball is None:
        return False
    if not state.ball_tracked:
        return False
    if state.ball.confidence < confidence_min:
        return False
    if not state.opponents:
        return False
    return (
        state.dist_ctrl_to_ball_carrier_normalized
        < state.tackle_carrier_distance_threshold
    )


def can_stand_tackle(state: MatchState, confidence_min: float = 0.85) -> bool:
    return can_tackle(state, confidence_min=confidence_min)


def can_sliding_tackle(state: MatchState, confidence_min: float = 0.85) -> bool:
    return can_tackle(state, confidence_min=confidence_min)


def _ball_ok_for_offense(state: MatchState, min_conf: float = 0.65) -> bool:
    if state.ball is None or not state.ball_tracked:
        return False
    return state.ball.confidence >= min_conf


def can_pass(state: MatchState) -> bool:
    if state.controlled_player is None or state.is_paused or state.wait_for_input:
        return False
    if not _ball_ok_for_offense(state):
        return False
    return len(state.passing_lanes) > 0


def can_through_ball(state: MatchState) -> bool:
    if state.controlled_player is None or state.is_paused or state.wait_for_input:
        return False
    if not _ball_ok_for_offense(state):
        return False
    return any(lane.is_clear and lane.distance > 100 for lane in state.passing_lanes)


def can_shoot(state: MatchState) -> bool:
    if state.controlled_player is None or state.is_paused or state.wait_for_input:
        return False
    if not _ball_ok_for_offense(state):
        return False
    return state.shot_opportunity > 0.1


def can_dribble_to_goal(state: MatchState) -> bool:
    """Always allowed — scorer drives goalward even when ball is missing (live tuning)."""
    return True


def can_clear(state: MatchState) -> bool:
    if state.controlled_player is None or state.ball is None:
        return False
    return state.distance_to_ball < 60 and state.defensive_danger > 0.4





def can_switch_player(state: MatchState) -> bool:
    """L1/LB switch — only when we have the ball (build-up / attack)."""
    if len(state.teammates) <= 1 or state.wait_for_input:
        return False
    return state.possession == PossessionState.OWN_TEAM


def can_rs_switch(state: MatchState) -> bool:
    """Pro (right-stick) switch when defending or contesting — needs ball + target."""
    if (
        len(state.teammates) <= 1
        or state.controlled_player is None
        or state.ball is None
        or state.wait_for_input
    ):
        return False
    return state.possession != PossessionState.OWN_TEAM

def can_gk_rush(state: MatchState) -> bool:
    """Можно ли выводить вратаря."""
    from state.match_state import PossessionState
    return len(state.goalkeepers_team) > 0 and state.possession != PossessionState.OWN_TEAM

def can_power_shot(state: MatchState) -> bool:
    """Условие для мощного удара."""
    return state.shot_opportunity > 0.3 and state.nearest_opponent_dist > 150 # Нужно пространство


def can_cancel(_state: MatchState) -> bool:
    return True

def can_contain(state: MatchState) -> bool:
    from state.match_state import PossessionState # Импортируем Enum
    return state.controlled_player is not None and state.possession != PossessionState.OWN_TEAM


# Map action types to their precondition functions
PRECONDITIONS: dict[ActionType, type] = {}  # populated at module level

_PRECONDITION_MAP = {
    ActionType.MOVE: can_move,
    ActionType.SPRINT: can_sprint,
    ActionType.JOCKEY: can_jockey,
    ActionType.STAND_TACKLE: can_stand_tackle,
    ActionType.SLIDING_TACKLE: can_sliding_tackle,
    ActionType.PASS: can_pass,
    ActionType.THROUGH_BALL: can_through_ball,
    ActionType.SHOOT: can_shoot,
    ActionType.CLEAR: can_clear,
    ActionType.CONTAIN: can_contain,
    ActionType.SWITCH_PLAYER: can_switch_player,
    ActionType.CANCEL: can_cancel,
    # FC26 compound actions share preconditions with their base actions
    ActionType.PRECISION_PASS: can_pass,
    ActionType.LOW_DRIVEN_SHOT: can_shoot,
    ActionType.FINESSE_SHOT: can_shoot,
    ActionType.TRIVELA_SHOT: can_shoot,
    ActionType.R1_DRIBBLE: can_move,
    ActionType.EXPLOSIVE_BURST: can_sprint,
    ActionType.L2_SHIELD: can_move,
    ActionType.SPEED_JOCKEY: can_jockey,
    ActionType.SUPER_CANCEL: can_cancel,
    ActionType.RS_SWITCH: can_rs_switch,
    ActionType.GK_NUDGE: can_move,
    ActionType.GK_RUSH: can_gk_rush,
    ActionType.DRIBBLE_TO_GOAL: can_dribble_to_goal,
}


def check_precondition(action: ActionType, state: MatchState) -> bool:
    """Check if an action is legal given the current state."""
    fn = _PRECONDITION_MAP.get(action)
    if fn is None:
        return False
    return fn(state)