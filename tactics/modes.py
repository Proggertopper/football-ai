"""Tactical finite-state machine.

Manages high-level tactical modes (build-up, counterattack, emergency
defense, etc.) and transitions between them based on match state scoring.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from config.settings import Config
from state.match_state import MatchState


class TacticalMode(enum.Enum):
    BUILD_UP = "build_up"
    SAFE_PROGRESSION = "safe_progression"
    FINAL_THIRD = "final_third"
    EMERGENCY_DEFENSE = "emergency_defense"
    PRESS_CONTAINMENT = "press_containment"
    BOX_DEFENSE = "box_defense"
    TRANSITION_RECOVERY = "transition_recovery"
    COUNTERATTACK = "counterattack"
    SET_PIECE = "set_piece"


@dataclass
class TacticalContext:
    """Extra context the tactical FSM provides to the decision engine."""

    mode: TacticalMode
    risk_tolerance: float = 0.5
    preferences: dict[str, float] = field(default_factory=dict)
    frames_in_mode: int = 0


class TacticalFSM:
    """Finite-state machine that selects the current tactical mode.

    Each frame, the evaluator scores every mode and the FSM transitions
    to the highest-scoring mode (subject to a cooldown to prevent rapid
    oscillation).
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg

        from tactics.evaluator import TacticalEvaluator
        self._evaluator = TacticalEvaluator()

        tact_cfg = getattr(cfg, "tactics", None)
        self._cooldown_max: int = getattr(tact_cfg, "transition_cooldown_frames", 30)
        self._cooldown: int = 0

        default_name = getattr(tact_cfg, "default_mode", "build_up")

        try:
            self._current = TacticalMode(default_name)
        except ValueError:
            self._current = TacticalMode.BUILD_UP

        self._frames_in_mode: int = 0
        self._mode_configs: dict[str, dict[str, float]] = {}

        if tact_cfg is not None and hasattr(tact_cfg, "modes"):
            modes_ns = tact_cfg.modes
            for mode in TacticalMode:
                if hasattr(modes_ns, mode.value):
                    section = getattr(modes_ns, mode.value)
                    raw = section.to_dict() if hasattr(section, "to_dict") else {}

                    cleaned: dict[str, float] = {}
                    for k, v in raw.items():
                        try:
                            cleaned[str(k)] = float(v)
                        except (TypeError, ValueError):
                            continue

                    self._mode_configs[mode.value] = cleaned

    def update(self, state: MatchState) -> TacticalContext:
        """Evaluate the state and potentially transition to a new mode."""
        self._frames_in_mode += 1

        if self._cooldown > 0:
            self._cooldown -= 1
        else:
            scores = self._evaluator.score_modes(state)
            best_mode = max(scores, key=scores.get)  # type: ignore[arg-type]
            if best_mode != self._current:
                self._current = best_mode
                self._frames_in_mode = 0
                self._cooldown = self._cooldown_max

        current_mode_key: str = str(self._current.value)

        mode_cfg = self._mode_configs.get(current_mode_key)
        if mode_cfg is None:
            mode_cfg = {}

        risk_raw = mode_cfg.get("risk_tolerance", 0.5)
        risk_tolerance = float(risk_raw)

        preferences: dict[str, float] = {
            k: float(v)
            for k, v in mode_cfg.items()
            if k != "risk_tolerance"
        }

        return TacticalContext(
            mode=self._current,
            risk_tolerance=risk_tolerance,
            preferences=preferences,
            frames_in_mode=self._frames_in_mode,
        )

    @property
    def current_mode(self) -> TacticalMode:
        return self._current

    def force_mode(self, mode: TacticalMode) -> None:
        """Force a mode transition (for testing or set-piece triggers)."""
        self._current = mode
        self._frames_in_mode = 0
        self._cooldown = self._cooldown_max

    def reset(self) -> None:
        self._current = TacticalMode.BUILD_UP
        self._frames_in_mode = 0
        self._cooldown = 0