"""Rule-based safety layer.

Filters and overrides action candidates to prevent dangerous decisions.
Runs BEFORE the final action selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config.settings import Config
from decision.actions import ActionSpec, ActionType
from state.match_state import MatchState, PossessionState, Zone


@dataclass
class SafetyFilterOutcome:
    """Result of safety filtering with explicit rejection reasons (offline debug)."""

    kept: list[ActionSpec]
    rejected: list[tuple[str, str]] = field(default_factory=list)  # (action.value, reason_code)


class SafetyLayer:
    """Applies hard safety rules to the candidate action list.

    Rules:
    1. Block actions at or above the max risk threshold (inclusive).
    2. Force clear/contain ordering in emergency defensive situations.
    3. Prevent shooting from non-attacking third.
    4. Prevent through balls under heavy pressure.
    5. Optional super-cancel on stumble (when such a candidate exists).
    """

    def __init__(self, cfg: Config) -> None:
        d_cfg = getattr(cfg, "decision", cfg)
        self._max_risk: float = float(getattr(d_cfg, "max_risk_threshold", 0.8))
        fc26 = getattr(cfg, "fc26", None)
        self._super_cancel_stumble: bool = bool(getattr(fc26, "super_cancel_on_stumble", True))

    def filter(self, candidates: list[ActionSpec], state: MatchState) -> SafetyFilterOutcome:
        """Return kept candidates plus rejection trace."""
        filtered: list[ActionSpec] = []
        rejected: list[tuple[str, str]] = []

        block_set_piece = state.wait_for_input and not getattr(
            state, "set_piece_wait_suppressed", False
        )

        for spec in candidates:
            if spec.risk >= self._max_risk:
                rejected.append((spec.action_type.value, "risk_threshold"))
                continue

            if block_set_piece and spec.action_type in (
                ActionType.PASS,
                ActionType.PRECISION_PASS,
                ActionType.THROUGH_BALL,
                ActionType.SHOOT,
                ActionType.LOW_DRIVEN_SHOT,
                ActionType.FINESSE_SHOT,
                ActionType.TRIVELA_SHOT,
                ActionType.SPRINT,
                ActionType.EXPLOSIVE_BURST,
                ActionType.RS_SWITCH,
                ActionType.SWITCH_PLAYER,
                ActionType.DRIBBLE_TO_GOAL,
            ):
                rejected.append((spec.action_type.value, "set_piece_wait"))
                continue

            if state.is_paused and spec.action_type in (
                ActionType.PASS,
                ActionType.PRECISION_PASS,
                ActionType.THROUGH_BALL,
                ActionType.SHOOT,
                ActionType.LOW_DRIVEN_SHOT,
                ActionType.FINESSE_SHOT,
                ActionType.TRIVELA_SHOT,
                ActionType.SPRINT,
                ActionType.EXPLOSIVE_BURST,
                ActionType.RS_SWITCH,
                ActionType.DRIBBLE_TO_GOAL,
            ):
                rejected.append((spec.action_type.value, "ball_lost_pause"))
                continue

            if spec.action_type in (
                ActionType.SHOOT,
                ActionType.LOW_DRIVEN_SHOT,
                ActionType.FINESSE_SHOT,
                ActionType.TRIVELA_SHOT,
            ):
                if state.controlled_zone != Zone.ATTACKING_THIRD:
                    rejected.append((spec.action_type.value, "non_attacking_shot"))
                    continue

            if spec.action_type == ActionType.THROUGH_BALL and state.local_pressure > 0.75:
                rejected.append((spec.action_type.value, "through_under_pressure"))
                continue

            if spec.action_type == ActionType.GK_RUSH:
                if not self._gk_rush_allowed(state):
                    rejected.append((spec.action_type.value, "gk_rush_context"))
                    continue

            filtered.append(spec)

        if state.defensive_danger > 0.8 and state.possession != PossessionState.OWN_TEAM:
            defensive_actions = {
                ActionType.CLEAR,
                ActionType.CONTAIN,
                ActionType.JOCKEY,
                ActionType.SPEED_JOCKEY,
                ActionType.STAND_TACKLE,
                ActionType.SLIDING_TACKLE,
            }
            defensive = [s for s in filtered if s.action_type in defensive_actions]
            if defensive:
                others = [s for s in filtered if s.action_type not in defensive_actions]
                filtered = defensive + others

        if self._super_cancel_stumble and _detect_stumble(state):
            super_cancel = [s for s in filtered if s.action_type == ActionType.SUPER_CANCEL]
            if super_cancel:
                return SafetyFilterOutcome(kept=super_cancel, rejected=rejected)

        if not filtered:
            filtered = self._fallback_specs(state)
            rejected.append(("*", "all_candidates_blocked_fallback"))

        return SafetyFilterOutcome(kept=filtered, rejected=rejected)

    @staticmethod
    def _gk_rush_allowed(state: MatchState) -> bool:
        """Relaxed vs. earlier version: only block obvious bad rush moments.

        Removed redundant ``distance_to_ball < 150`` check — without a controlled
        field player that distance is not meaningful for GK logic and suppressed
        nearly all GK_RUSH attempts in offline tests.
        """
        if state.nearest_opponent_dist_to_gk < 80:
            return False
        return True

    @staticmethod
    def _fallback_specs(state: MatchState) -> list[ActionSpec]:
        """Prefer a neutral MOVE (debuggable) over CANCEL so the pipeline keeps moving."""
        if state.controlled_player is not None:
            return [
                ActionSpec(
                    action_type=ActionType.MOVE,
                    score=0.01,
                    risk=0.0,
                    expected_value=0.01,
                    direction=(1.0, 0.0),
                    duration_frames=4,
                    interruptible=True,
                    metadata={"fallback": 1.0},
                )
            ]
        return [
            ActionSpec(
                action_type=ActionType.CANCEL,
                score=0.01,
                risk=0.0,
                expected_value=0.0,
                metadata={"fallback": 1.0},
            )
        ]


def _detect_stumble(state: MatchState) -> bool:
    """Heuristic stumble detection: sudden velocity drop after contact."""
    if state.controlled_player is None or state.previous is None:
        return False
    prev = state.previous
    if prev.controlled_player is None:
        return False

    cur_speed = state.controlled_player.speed
    prev_speed = prev.controlled_player.speed

    if prev_speed > 3.0 and cur_speed < prev_speed * 0.4:
        if state.nearest_opponent_dist < 80:
            return True
    return False
