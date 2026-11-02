"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

import pytest

from config.settings import Config, load_config
from decision.engine import DecisionEngine
from evaluate.fixtures import (
    fixture_counterattack,
    fixture_crowded_box,
    fixture_open_shot,
    fixture_safe_pass_left,
    fixture_under_pressure_defense,
)
from state.estimator import StateEstimator
from state.match_state import MatchState
from vision.types import  TrackedObject


@pytest.fixture
def cfg() -> Config:
    """Load the default config."""
    return load_config()


@pytest.fixture
def state_estimator(cfg: Config) -> StateEstimator:
    return StateEstimator(cfg)


@pytest.fixture
def decision_engine(cfg: Config) -> DecisionEngine:
    return DecisionEngine(cfg)


@pytest.fixture
def open_shot_tracked() -> list[TrackedObject]:
    return fixture_open_shot()


@pytest.fixture
def safe_pass_tracked() -> list[TrackedObject]:
    return fixture_safe_pass_left()


@pytest.fixture
def defense_tracked() -> list[TrackedObject]:
    return fixture_under_pressure_defense()


@pytest.fixture
def counter_tracked() -> list[TrackedObject]:
    return fixture_counterattack()


@pytest.fixture
def crowded_tracked() -> list[TrackedObject]:
    return fixture_crowded_box()


@pytest.fixture
def open_shot_state(
        state_estimator: StateEstimator, open_shot_tracked: list[TrackedObject]
) -> MatchState:
    return state_estimator.estimate(open_shot_tracked, frame_id=1)


@pytest.fixture
def defense_state(
        state_estimator: StateEstimator, defense_tracked: list[TrackedObject]
) -> MatchState:
    return state_estimator.estimate(defense_tracked, frame_id=1)


@pytest.fixture
def counter_state(
        state_estimator: StateEstimator, counter_tracked: list[TrackedObject]
) -> MatchState:
    return state_estimator.estimate(counter_tracked, frame_id=1)