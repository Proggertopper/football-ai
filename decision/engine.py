"""Hybrid decision engine.

Pipeline:
1. Scorer produces candidates (heuristic evaluation of every legal action)
2. Safety layer filters out dangerous actions and enforces emergency overrides
3. Hysteresis filter stabilizes the final selection across frames

The engine also exposes an RL-ready interface: given a state, return
the scored action list so an external policy can override the selection.
"""

from __future__ import annotations

from dataclasses import dataclass

from config.settings import Config
from decision.actions import ActionSpec, ActionType
from decision.hysteresis import HysteresisFilter
from decision.pipeline_debug import log_decision_trace
from decision.scorer import ActionScorer
from state.match_state import MatchState


@dataclass
class DecisionResult:
    """Output of the decision engine for one frame."""

    chosen: ActionSpec
    candidates: list[ActionSpec]  # all scored candidates (pre-filter)
    filtered: list[ActionSpec]  # post-safety-filter
    frame_id: int = 0


class DecisionEngine:
    """Main decision pipeline.

    Usage:
        engine = DecisionEngine(cfg)
        result = engine.decide(state)
        # result.chosen is the ActionSpec to execute
    """

    def __init__(self, cfg: Config) -> None:
        self._scorer = ActionScorer(cfg)
        self._hysteresis = HysteresisFilter(cfg)
        obs = getattr(cfg, "observe", None)
        self._trace_verbose: bool = bool(getattr(obs, "decision_trace_verbose", False))

    @property
    def scorer(self):
        """Легальный доступ к скореру для дебага."""
        return self._scorer

    def decide(self, state: MatchState) -> DecisionResult:
        """Run the full decision pipeline and return the chosen action."""
        candidates = self._scorer.score_all(state)
        # Safety layer disabled — keep full candidate list for debugging / RL parity
        filtered = list(candidates)
        safety_out_rejected: list[tuple[str, str]] = []

        chosen = self._hysteresis.apply(filtered)

        self._scorer.apply_cooldown(chosen.action_type)

        log_decision_trace(
            state=state,
            candidates=candidates,
            filtered=filtered,
            chosen=chosen,
            rejected=safety_out_rejected,
            skipped=self._scorer.last_skipped,
            verbose=self._trace_verbose,
        )

        return DecisionResult(
            chosen=chosen,
            candidates=candidates,
            filtered=filtered,
            frame_id=state.frame_id,
        )

    def get_scored_actions(self, state: MatchState) -> list[ActionSpec]:
        """RL-ready interface: return all scored actions without selecting.

        An external policy (imitation learning, RL agent) can use this
        to observe the heuristic scores and override the selection.
        """
        return list(self._scorer.score_all(state))

    def force_action(self, action_type: ActionType, state: MatchState) -> DecisionResult:
        """Force a specific action (for testing or manual override)."""
        candidates = self._scorer.score_all(state)
        forced = None
        for spec in candidates:
            if spec.action_type == action_type:
                forced = spec
                break
        if forced is None:
            forced = ActionSpec(
                action_type=action_type,
                score=0.0,
                risk=0.0,
                expected_value=0.0,
            )
        return DecisionResult(
            chosen=forced,
            candidates=candidates,
            filtered=candidates,
            frame_id=state.frame_id,
        )

    def reset(self) -> None:
        """Reset engine state (hysteresis, cooldowns)."""
        self._hysteresis.force_reset()
        self._scorer.reset_cooldowns()

    def reset_hysteresis(self) -> None:
        """Clear hysteresis only (e.g. when pausing live bot — avoids stale hold state)."""
        self._hysteresis.force_reset()