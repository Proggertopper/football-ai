"""Compact per-frame decision tracing for offline debugging."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from decision.actions import ActionSpec
    from state.match_state import MatchState

logger = logging.getLogger("football_ai.decision")


def log_decision_trace(
    *,
    state: MatchState,
    candidates: list[ActionSpec],
    filtered: list[ActionSpec],
    chosen: ActionSpec,
    rejected: list[tuple[str, str]],
    skipped: list[tuple[str, str]],
    verbose: bool = False,
) -> None:
    """Log one structured line + optional detail (verbose).

    *rejected* / *skipped* use (action_name, reason_code).
    """
    ctrl_id = state.controlled_player.track_id if state.controlled_player else None
    ball_id = state.ball.track_id if state.ball else None
    top_pre = ", ".join(
        f"{c.action_type.value}:{c.score:.2f}" for c in candidates[:5]
    )
    top_post = ", ".join(
        f"{c.action_type.value}:{c.score:.2f}" for c in filtered[:5]
    )
    msg = (
        f"trace frame={state.frame_id} possession={state.possession.value} "
        f"ctrl_track={ctrl_id} ball_track={ball_id} "
        f"dist_ball={state.distance_to_ball if state.distance_to_ball < 1e8 else -1.0:.1f} "
        f"pre=[{top_pre or '—'}] post=[{top_post or '—'}] "
        f"chosen={chosen.action_type.value} ev={chosen.expected_value:.2f} risk={chosen.risk:.2f}"
    )
    if verbose:
        logger.info(msg)
    else:
        logger.debug(msg)
    if verbose:
        for action, code in rejected:
            logger.info("trace frame=%s rejected %s: %s", state.frame_id, action, code)
        for action, code in skipped:
            logger.info("trace frame=%s skipped %s: %s", state.frame_id, action, code)
    else:
        for action, code in rejected:
            logger.debug("trace frame=%s rejected %s: %s", state.frame_id, action, code)
        for action, code in skipped:
            logger.debug("trace frame=%s skipped %s: %s", state.frame_id, action, code)


def format_skipped_actions(skipped: list[tuple[str, str]]) -> dict[str, Any]:
    """Serialize skip list for replay / JSON overlays."""
    return {"skipped": [{"action": a, "reason": r} for a, r in skipped]}
