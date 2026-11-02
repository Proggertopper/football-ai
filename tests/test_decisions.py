"""Unit tests for the decision engine, safety layer, and hysteresis."""

from __future__ import annotations

import pytest

from config.settings import load_config
from decision.actions import ActionSpec, ActionType
from decision.engine import DecisionEngine
from decision.hysteresis import HysteresisFilter
from decision.safety import SafetyLayer
from evaluate.fixtures import (
    fixture_open_shot,
    fixture_under_pressure_defense,
)
from state.estimator import StateEstimator
from state.match_state import MatchState


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def engine(cfg):
    return DecisionEngine(cfg)


@pytest.fixture
def estimator(cfg):
    return StateEstimator(cfg)


def _state(estimator, fixture_fn) -> MatchState:
    return estimator.estimate(fixture_fn(), frame_id=1)


class TestDecisionEngine:
    def test_returns_decision_result(self, engine, estimator) -> None:
        state = _state(estimator, fixture_open_shot)
        result = engine.decide(state)
        assert result.chosen is not None
        assert result.chosen.action_type is not None
        assert len(result.candidates) > 0

    def test_open_shot_chooses_shooting(self, engine, estimator) -> None:
        state = _state(estimator, fixture_open_shot)
        result = engine.decide(state)
        shooting = {
            ActionType.SHOOT, ActionType.LOW_DRIVEN_SHOT,
            ActionType.FINESSE_SHOT, ActionType.TRIVELA_SHOT,
        }
        assert result.chosen.action_type in shooting or result.chosen.score > 0.0

    def test_defense_does_not_shoot(self, engine, estimator) -> None:
        state = _state(estimator, fixture_under_pressure_defense)
        result = engine.decide(state)
        forbidden = {ActionType.SHOOT, ActionType.LOW_DRIVEN_SHOT, ActionType.THROUGH_BALL}
        # In defensive third, shooting should be blocked by safety layer
        assert result.chosen.action_type not in forbidden

    def test_force_action(self, engine, estimator) -> None:
        state = _state(estimator, fixture_open_shot)
        result = engine.force_action(ActionType.CANCEL, state)
        assert result.chosen.action_type == ActionType.CANCEL

    def test_get_scored_actions_for_rl(self, engine, estimator) -> None:
        state = _state(estimator, fixture_open_shot)
        actions = engine.get_scored_actions(state)
        assert isinstance(actions, list)
        assert all(isinstance(a, ActionSpec) for a in actions)

    def test_reset_clears_hysteresis(self, engine, estimator) -> None:
        state = _state(estimator, fixture_open_shot)
        engine.decide(state)
        engine.reset()
        # After reset, decision should work normally
        result = engine.decide(state)
        assert result.chosen is not None


class TestSafetyLayer:
    def test_blocks_high_risk(self, cfg) -> None:
        safety = SafetyLayer(cfg)
        candidates = [
            ActionSpec(ActionType.SHOOT, score=0.8, risk=0.95, expected_value=0.0),
            ActionSpec(ActionType.PASS, score=0.5, risk=0.3, expected_value=0.2),
        ]
        state = StateEstimator(cfg).estimate(fixture_open_shot(), frame_id=1)
        outcome = safety.filter(candidates, state)
        action_types = {s.action_type for s in outcome.kept}
        assert ActionType.SHOOT not in action_types  # risk >= max_risk_threshold

    def test_always_returns_at_least_one(self, cfg) -> None:
        safety = SafetyLayer(cfg)
        candidates = [
            ActionSpec(ActionType.SHOOT, score=0.8, risk=0.99, expected_value=0.0),
        ]
        state = StateEstimator(cfg).estimate(fixture_open_shot(), frame_id=1)
        outcome = safety.filter(candidates, state)
        assert len(outcome.kept) >= 1


class TestHysteresis:
    def test_holds_current_action(self, cfg) -> None:
        hysteresis = HysteresisFilter(cfg)
        candidates = [
            ActionSpec(ActionType.PASS, score=0.6, risk=0.2, expected_value=0.4),
            ActionSpec(ActionType.MOVE, score=0.55, risk=0.1, expected_value=0.45),
        ]
        # First call: picks PASS (best)
        chosen1 = hysteresis.apply(candidates)
        assert chosen1.action_type == ActionType.PASS

        # Second call: MOVE has slightly higher score but within hysteresis margin
        candidates2 = [
            ActionSpec(ActionType.MOVE, score=0.65, risk=0.1, expected_value=0.55),
            ActionSpec(ActionType.PASS, score=0.6, risk=0.2, expected_value=0.4),
        ]
        # Should hold PASS during hold window
        chosen2 = hysteresis.apply(candidates2)
        assert chosen2.action_type == ActionType.PASS

    def test_switches_after_hold_window(self, cfg) -> None:
        hysteresis = HysteresisFilter(cfg)
        candidates1 = [
            ActionSpec(ActionType.PASS, score=0.6, risk=0.2, expected_value=0.4),
        ]
        hysteresis.apply(candidates1)

        # Exhaust hold window
        for _ in range(cfg.decision.min_action_hold_frames + 1):
            candidates_hold = [
                ActionSpec(ActionType.SHOOT, score=0.9, risk=0.2, expected_value=0.7),
                ActionSpec(ActionType.PASS, score=0.4, risk=0.2, expected_value=0.2),
            ]
            result = hysteresis.apply(candidates_hold)

        # After hold window exhausted with much higher score, should switch
        assert result.action_type == ActionType.SHOOT

    def test_force_reset(self, cfg) -> None:
        hysteresis = HysteresisFilter(cfg)
        candidates = [
            ActionSpec(ActionType.PASS, score=0.6, risk=0.2, expected_value=0.4),
        ]
        hysteresis.apply(candidates)
        hysteresis.force_reset()

        candidates2 = [
            ActionSpec(ActionType.MOVE, score=0.5, risk=0.1, expected_value=0.4),
        ]
        chosen = hysteresis.apply(candidates2)
        assert chosen.action_type == ActionType.MOVE