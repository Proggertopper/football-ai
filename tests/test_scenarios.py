"""Scenario-based integration tests using the evaluation harness.

These tests run full scenarios through estimation → decision and
verify that the chosen action is acceptable for each situation.
"""

from __future__ import annotations

import pytest

from config.settings import load_config
from evaluate.harness import EvaluationHarness
from evaluate.scenarios import SCENARIOS, get_scenario


@pytest.fixture
def harness() -> EvaluationHarness:
    return EvaluationHarness(load_config())


class TestScenarioHarness:
    def test_all_scenarios_run_without_error(self, harness: EvaluationHarness) -> None:
        """Smoke test: every scenario completes without exceptions."""
        report = harness.run_all()
        assert report.total == len(SCENARIOS)
        assert report.total > 0

    def test_open_shot_scenario(self, harness: EvaluationHarness) -> None:
        scenario = get_scenario("open_shot")
        assert scenario is not None
        result = harness.run_one(scenario)
        assert result.passed, f"Failed: {result.reason} (chose {result.chosen_action})"

    def test_blocked_shot_scenario(self, harness: EvaluationHarness) -> None:
        scenario = get_scenario("blocked_shot")
        assert scenario is not None
        result = harness.run_one(scenario)
        assert result.passed, f"Failed: {result.reason} (chose {result.chosen_action})"

    def test_safe_pass_scenario(self, harness: EvaluationHarness) -> None:
        scenario = get_scenario("safe_pass_left")
        assert scenario is not None
        result = harness.run_one(scenario)
        assert result.passed, f"Failed: {result.reason} (chose {result.chosen_action})"

    def test_defense_scenario(self, harness: EvaluationHarness) -> None:
        scenario = get_scenario("under_pressure_defense")
        assert scenario is not None
        result = harness.run_one(scenario)
        assert result.passed, f"Failed: {result.reason} (chose {result.chosen_action})"

    def test_loose_ball_scenario(self, harness: EvaluationHarness) -> None:
        scenario = get_scenario("loose_ball_race")
        assert scenario is not None
        result = harness.run_one(scenario)
        assert result.passed, f"Failed: {result.reason} (chose {result.chosen_action})"

    def test_crowded_box_scenario(self, harness: EvaluationHarness) -> None:
        scenario = get_scenario("crowded_box")
        assert scenario is not None
        result = harness.run_one(scenario)
        assert result.passed, f"Failed: {result.reason} (chose {result.chosen_action})"

    def test_counterattack_scenario(self, harness: EvaluationHarness) -> None:
        scenario = get_scenario("counterattack")
        assert scenario is not None
        result = harness.run_one(scenario)
        assert result.passed, f"Failed: {result.reason} (chose {result.chosen_action})"

    def test_report_summary_format(self, harness: EvaluationHarness) -> None:
        report = harness.run_all()
        summary = report.summary()
        assert "Evaluation Report" in summary
        assert "PASS" in summary or "FAIL" in summary

    def test_scenario_timing(self, harness: EvaluationHarness) -> None:
        """Each scenario should complete in under 100ms (no model loaded)."""
        report = harness.run_all()
        for result in report.results:
            assert result.elapsed_ms < 100, (
                f"Scenario {result.scenario_name} took {result.elapsed_ms:.1f}ms"
            )