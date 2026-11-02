"""Unit tests for the action scorer."""

from __future__ import annotations

import pytest

from typing import Callable as Callable


from config.settings import  load_config
from decision.actions import ActionType
from decision.scorer import ActionScorer
from state.estimator import StateEstimator
from state.match_state import MatchState
from evaluate.fixtures import (
    fixture_open_shot,
    fixture_safe_pass_left,
    fixture_under_pressure_defense,
)


@pytest.fixture
def scorer() -> ActionScorer:
    cfg = load_config()
    return ActionScorer(cfg)


@pytest.fixture
def estimator() -> StateEstimator:
    cfg = load_config()
    return StateEstimator(cfg)


def _build_state(estimator: StateEstimator, fixture_fn: Callable) -> MatchState:
    tracked = fixture_fn()
    return estimator.estimate(tracked, frame_id=1)


class TestActionScorer:
    def test_open_shot_prefers_shooting(
            self, scorer: ActionScorer, estimator: StateEstimator
    ) -> None:
        state = _build_state(estimator, fixture_open_shot)
        candidates = scorer.score_all(state)
        assert len(candidates) > 0
        # At least one shooting action should be in top 5
        top_actions = {c.action_type for c in candidates[:5]}
        shooting_actions = {
            ActionType.SHOOT, ActionType.LOW_DRIVEN_SHOT,
            ActionType.FINESSE_SHOT, ActionType.TRIVELA_SHOT,
        }
        assert top_actions & shooting_actions, (
            f"No shooting action in top 5: {[c.action_type.value for c in candidates[:5]]}"
        )

    def test_safe_pass_prefers_passing(
            self, scorer: ActionScorer, estimator: StateEstimator
    ) -> None:
        state = _build_state(estimator, fixture_safe_pass_left)
        candidates = scorer.score_all(state)
        # Passing must stay competitive vs. dribble/sprint (tactical stack ranks several actions high).
        top_actions = {c.action_type for c in candidates[:12]}
        passing_actions = {
            ActionType.PASS, ActionType.PRECISION_PASS, ActionType.THROUGH_BALL,
        }
        assert top_actions & passing_actions

    def test_defense_prefers_defensive_actions(
            self, scorer: ActionScorer, estimator: StateEstimator
    ) -> None:
        state = _build_state(estimator, fixture_under_pressure_defense)
        candidates = scorer.score_all(state)
        assert len(candidates) > 0
        top_actions = {c.action_type for c in candidates[:5]}
        defensive = {
            ActionType.CLEAR, ActionType.CONTAIN, ActionType.JOCKEY,
            ActionType.SPEED_JOCKEY, ActionType.STAND_TACKLE,
        }
        assert top_actions & defensive

    def test_cooldown_blocks_action(
            self, scorer: ActionScorer, estimator: StateEstimator
    ) -> None:
        state = _build_state(estimator, fixture_open_shot)
        scorer.apply_cooldown(ActionType.SHOOT)
        candidates = scorer.score_all(state)
        # SHOOT should not be in the list while on cooldown
        action_types = {c.action_type for c in candidates}
        assert ActionType.SHOOT not in action_types

    def test_all_candidates_have_scores(
            self, scorer: ActionScorer, estimator: StateEstimator
    ) -> None:
        state = _build_state(estimator, fixture_open_shot)
        candidates = scorer.score_all(state)
        for c in candidates:
            assert isinstance(c.score, float)
            assert isinstance(c.risk, float)
            assert isinstance(c.expected_value, float)

    def test_candidates_sorted_descending(
            self, scorer: ActionScorer, estimator: StateEstimator
    ) -> None:
        state = _build_state(estimator, fixture_open_shot)
        candidates = scorer.score_all(state)
        scores = [c.score for c in candidates]
        assert scores == sorted(scores, reverse=True)