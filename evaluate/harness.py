"""Offline evaluation harness.

Runs scenarios through the full pipeline (state estimation → decision)
without requiring a live game, screen capture, or gamepad. Reports
pass/fail for each scenario and produces a summary.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from config.settings import Config, load_config
from decision.engine import DecisionEngine, DecisionResult
from evaluate.scenarios import SCENARIOS, Scenario
from state.estimator import StateEstimator
from state.match_state import MatchState
from vision.types import TrackedObject


@dataclass
class ScenarioResult:
    """Result of running one scenario."""

    scenario_name: str
    passed: bool
    chosen_action: str
    chosen_score: float
    chosen_risk: float
    reason: str = ""
    elapsed_ms: float = 0.0
    all_candidates: list[str] = field(default_factory=list)


@dataclass
class EvaluationReport:
    """Summary of all scenario results."""

    results: list[ScenarioResult]
    total: int = 0
    passed: int = 0
    failed: int = 0
    elapsed_ms: float = 0.0

    def summary(self) -> str:
        lines = [
            f"{'=' * 60}",
            f"  Evaluation Report: {self.passed}/{self.total} passed"
            f"  ({self.elapsed_ms:.1f}ms total)",
            f"{'=' * 60}",
        ]
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(
                f"  [{status}] {r.scenario_name:30s} → {r.chosen_action:20s} "
                f"(score={r.chosen_score:.3f}, risk={r.chosen_risk:.3f})"
            )
            if not r.passed:
                lines.append(f"         Reason: {r.reason}")
        lines.append(f"{'=' * 60}")
        return "\n".join(lines)


class EvaluationHarness:
    """Runs scenarios and produces an EvaluationReport.

    Usage:
        harness = EvaluationHarness()
        report = harness.run_all()
        print(report.summary())
    """

    def __init__(self, cfg: Config | None = None) -> None:
        self._cfg = cfg or load_config()
        self._estimator = StateEstimator(self._cfg)
        self._engine = DecisionEngine(self._cfg)

    def run_all(self, scenarios: list[Scenario] | None = None) -> EvaluationReport:
        """Run all scenarios (or a provided subset) and return the report."""
        target = scenarios or SCENARIOS
        results: list[ScenarioResult] = []
        start = time.perf_counter()

        for scenario in target:
            result = self.run_one(scenario)
            results.append(result)

        elapsed = (time.perf_counter() - start) * 1000
        passed = sum(1 for r in results if r.passed)
        return EvaluationReport(
            results=results,
            total=len(results),
            passed=passed,
            failed=len(results) - passed,
            elapsed_ms=elapsed,
        )

    def run_one(self, scenario: Scenario) -> ScenarioResult:
        """Run a single scenario through estimation → decision."""
        # Fresh estimator and engine per scenario to avoid state leakage
        self._estimator = StateEstimator(self._cfg)
        self._engine = DecisionEngine(self._cfg)

        t0 = time.perf_counter()

        # Build tracked objects from fixture
        tracked: list[TrackedObject] = scenario.fixture_fn()

        # Estimate state
        state: MatchState = self._estimator.estimate(tracked, frame_id=0)

        # Decide
        decision: DecisionResult = self._engine.decide(state)
        chosen = decision.chosen
        elapsed = (time.perf_counter() - t0) * 1000

        # Evaluate pass/fail
        passed = True
        reason = ""

        if chosen.action_type not in scenario.acceptable_actions:
            passed = False
            acceptable = ", ".join(a.value for a in scenario.acceptable_actions)
            reason = f"Chose {chosen.action_type.value} but expected one of: {acceptable}"

        if chosen.action_type in scenario.forbidden_actions:
            passed = False
            reason = f"Chose forbidden action: {chosen.action_type.value}"

        if chosen.score < scenario.min_score:
            passed = False
            reason = f"Score {chosen.score:.3f} below minimum {scenario.min_score}"

        if chosen.risk > scenario.max_risk:
            passed = False
            reason = f"Risk {chosen.risk:.3f} above maximum {scenario.max_risk}"

        return ScenarioResult(
            scenario_name=scenario.name,
            passed=passed,
            chosen_action=chosen.action_type.value,
            chosen_score=chosen.score,
            chosen_risk=chosen.risk,
            reason=reason,
            elapsed_ms=elapsed,
            all_candidates=[c.action_type.value for c in decision.candidates[:5]],
        )