"""Unit tests for state estimation."""

from __future__ import annotations

import pytest

from config.settings import load_config
from evaluate.fixtures import (
    fixture_counterattack,
    fixture_open_shot,
    fixture_safe_pass_left,
    fixture_under_pressure_defense,
)
from state.estimator import StateEstimator
from state.match_state import PossessionState, Zone



@pytest.fixture
def estimator() -> StateEstimator:
    return StateEstimator(load_config())


class TestStateEstimation:
    def test_ball_detected(self, estimator: StateEstimator) -> None:
        state = estimator.estimate(fixture_open_shot(), frame_id=1)
        assert state.ball is not None
        assert state.ball.position.x > 0

    def test_controlled_player_identified(self, estimator: StateEstimator) -> None:
        state = estimator.estimate(fixture_open_shot(), frame_id=1)
        assert state.controlled_player is not None

    def test_teammates_classified(self, estimator: StateEstimator) -> None:
        state = estimator.estimate(fixture_open_shot(), frame_id=1)
        # fixture has 3 team players (one is controlled)
        assert len(state.teammates) >= 2

    def test_opponents_classified(self, estimator: StateEstimator) -> None:
        state = estimator.estimate(fixture_open_shot(), frame_id=1)
        assert len(state.opponents) >= 2

    def test_possession_own_team_when_close(self, estimator: StateEstimator) -> None:
        state = estimator.estimate(fixture_open_shot(), frame_id=1)
        assert state.possession == PossessionState.OWN_TEAM

    def test_zone_classification_attacking(self, estimator: StateEstimator) -> None:
        state = estimator.estimate(fixture_open_shot(), frame_id=1)
        # Ball at x=1700 on 1920 screen → attacking third
        assert state.ball_zone == Zone.ATTACKING_THIRD

    def test_zone_classification_defensive(self, estimator: StateEstimator) -> None:
        state = estimator.estimate(fixture_under_pressure_defense(), frame_id=1)
        assert state.ball_zone == Zone.DEFENSIVE_THIRD

    def test_pressure_high_in_defense(self, estimator: StateEstimator) -> None:
        state = estimator.estimate(fixture_under_pressure_defense(), frame_id=1)
        assert state.local_pressure > 0.3

    def test_pressure_low_in_open_shot(self, estimator: StateEstimator) -> None:
        state = estimator.estimate(fixture_open_shot(), frame_id=1)
        assert state.local_pressure < 0.5

    def test_defensive_danger_high(self, estimator: StateEstimator) -> None:
        state = estimator.estimate(fixture_under_pressure_defense(), frame_id=1)
        assert state.defensive_danger > 0.3

    def test_counterattack_opportunity(self, estimator: StateEstimator) -> None:
        state = estimator.estimate(fixture_counterattack(), frame_id=1)
        assert state.counterattack_opportunity > 0.2

    def test_passing_lanes_exist(self, estimator: StateEstimator) -> None:
        state = estimator.estimate(fixture_safe_pass_left(), frame_id=1)
        assert len(state.passing_lanes) > 0

    def test_shot_lanes_exist(self, estimator: StateEstimator) -> None:
        state = estimator.estimate(fixture_open_shot(), frame_id=1)
        assert len(state.shot_lanes) > 0

    def test_best_shot_lane_quality(self, estimator: StateEstimator) -> None:
        state = estimator.estimate(fixture_open_shot(), frame_id=1)
        assert state.best_shot_lane is not None
        assert state.best_shot_lane.quality > 0.0

    def test_predicted_positions(self, estimator: StateEstimator) -> None:
        state = estimator.estimate(fixture_open_shot(), frame_id=1)
        assert state.predicted_ball_pos.x != 0.0 or state.predicted_ball_pos.y != 0.0

    def test_history_linked(self, estimator: StateEstimator) -> None:
        estimator.estimate(fixture_open_shot(), frame_id=0)
        state2 = estimator.estimate(fixture_open_shot(), frame_id=1)
        assert state2.previous is not None
        assert state2.previous.frame_id == 0

    def test_distance_to_ball(self, estimator: StateEstimator) -> None:
        state = estimator.estimate(fixture_open_shot(), frame_id=1)
        assert state.distance_to_ball < 50  # controlled player is near ball