"""Tactical mode evaluator.

Scores every tactical mode based on match state features.
The FSM transitions to the highest-scoring mode.
"""

from __future__ import annotations

from state.match_state import MatchState, PossessionState, Zone
from tactics.modes import TacticalMode


class TacticalEvaluator:
    """Stateless evaluator — scores each tactical mode for the current frame."""

    def score_modes(self, state: MatchState) -> dict[TacticalMode, float]:
        """Return a score for each tactical mode (higher = more appropriate)."""
        scores: dict[TacticalMode, float] = {TacticalMode.BUILD_UP: self._score_build_up(state),
                                             TacticalMode.SAFE_PROGRESSION: self._score_safe_progression(state),
                                             TacticalMode.FINAL_THIRD: self._score_final_third(state),
                                             TacticalMode.EMERGENCY_DEFENSE: self._score_emergency_defense(state),
                                             TacticalMode.PRESS_CONTAINMENT: self._score_press_containment(state),
                                             TacticalMode.BOX_DEFENSE: self._score_box_defense(state),
                                             TacticalMode.TRANSITION_RECOVERY: self._score_transition_recovery(state),
                                             TacticalMode.COUNTERATTACK: self._score_counterattack(state),
                                             TacticalMode.SET_PIECE: self._score_set_piece(state)}
        return scores

    @staticmethod
    def _score_build_up(state: MatchState) -> float:
        score = 0.0
        if state.possession == PossessionState.OWN_TEAM:
            score += 0.4
        if state.controlled_zone == Zone.DEFENSIVE_THIRD:
            score += 0.3
        if state.local_pressure < 0.3:
            score += 0.2
        return score

    @staticmethod
    def _score_safe_progression(state: MatchState) -> float:
        score = 0.0
        if state.possession == PossessionState.OWN_TEAM:
            score += 0.3
        if state.controlled_zone == Zone.MIDDLE_THIRD:
            score += 0.3
        if state.local_pressure < 0.4:
            score += 0.2
        if state.defensive_danger < 0.3:
            score += 0.1
        return score

    @staticmethod
    def _score_final_third(state: MatchState) -> float:
        score = 0.0
        if state.possession == PossessionState.OWN_TEAM:
            score += 0.3
        if state.controlled_zone == Zone.ATTACKING_THIRD:
            score += 0.4
        if state.shot_opportunity > 0.3:
            score += 0.2
        return score

    @staticmethod
    def _score_emergency_defense(state: MatchState) -> float:
        score = 0.0
        if state.defensive_danger > 0.7:
            score += 0.5
        # --- НОВАЯ ЛОГИКА ---
        # Если враг совсем рядом с вратарем, это точно критическая ситуация
        if state.nearest_opponent_dist_to_gk < 100:
            score += 0.3
        # --------------------
        if state.possession == PossessionState.OPPONENT:
            score += 0.2
        return score

    @staticmethod
    def _score_press_containment(state: MatchState) -> float:
        score = 0.0
        if state.possession == PossessionState.OPPONENT:
            score += 0.3
        if state.ball_zone == Zone.MIDDLE_THIRD:
            score += 0.2
        if state.local_pressure < 0.5:
            score += 0.1
        # Good when we're organized but don't have the ball
        if state.defensive_danger < 0.5:
            score += 0.1
        return score

    @staticmethod
    def _score_box_defense(state: MatchState) -> float:
        score = 0.0
        if state.defensive_danger > 0.5:
            score += 0.3
        if state.ball_zone == Zone.DEFENSIVE_THIRD:
            score += 0.3
        if state.possession == PossessionState.OPPONENT:
            score += 0.2
        return score

    @staticmethod
    def _score_transition_recovery(state: MatchState) -> float:
        score = 0.0
        # Just lost possession — need to recover
        if state.possession in (PossessionState.OPPONENT, PossessionState.LOOSE):
            score += 0.2
        if state.previous is not None:
            if state.previous.possession == PossessionState.OWN_TEAM:
                if state.possession != PossessionState.OWN_TEAM:
                    score += 0.4  # turnover just happened
        if state.controlled_zone == Zone.ATTACKING_THIRD:
            score += 0.1  # caught high up the pitch
        return score

    @staticmethod
    def _score_counterattack(state: MatchState) -> float:
        score = state.counterattack_opportunity * 0.6
        if state.possession == PossessionState.OWN_TEAM:
            score += 0.2
        if state.dribble_space > 150:
            score += 0.15
        return score

    @staticmethod
    def _score_set_piece(_state: MatchState) -> float:
        # Placeholder: set-piece detection requires game-event integration.
        # For now this mode is only entered via force_mode().
        return 0.0