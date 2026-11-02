"""Anti-thrashing / hysteresis logic.

Prevents the AI from oscillating between actions every frame by requiring
a minimum score improvement before switching, and enforcing a minimum
hold duration for the current action.
"""

from __future__ import annotations

from config.settings import Config
from decision.actions import ActionSpec, ActionType


class HysteresisFilter:
    """Stabilizes action selection across frames.

    - The current action keeps a "loyalty bonus" equal to hysteresis_threshold.
    - An action must outscore the current action by at least that margin.
    - Once an action is selected, it holds for at least min_hold_frames.
    """
    def __init__(self, cfg: Config) -> None:
        # Пытаемся найти секцию 'decision', если её нет — ищем прямо в cfg
        d_cfg = getattr(cfg, 'decision', cfg)
        self._threshold: float = getattr(d_cfg, 'hysteresis_threshold', 0.1)
        self._min_hold: int = getattr(d_cfg, 'min_action_hold_frames', 3)
        self._current_action: ActionType | None = None
        self._hold_counter: int = 0

    def apply(self, candidates: list[ActionSpec]) -> ActionSpec:
        """Select the best action, applying hysteresis to the current choice."""
        if not candidates:
            return ActionSpec(
                action_type=ActionType.CANCEL,
                score=0.0,
                risk=0.0,
                expected_value=0.0,
            )

        best = candidates[0]

        # 1. Мгновенная реакция на CANCEL (ускоряет отклик в FC 26)
        if best.action_type == ActionType.CANCEL:
            self.force_reset()
            return best

        # 2. Удержание текущего действия (Hold window)
        if self._hold_counter > 0 and self._current_action is not None:
            current_candidate = self._find_action(candidates, self._current_action)
            if current_candidate is not None:
                self._hold_counter -= 1
                return current_candidate
            self._hold_counter = 0

        # 3. Loyalty bonus — keep current action if still competitive
        current_action = self._current_action

        if current_action is not None and current_action != best.action_type:
            current_candidate = self._find_action(candidates, current_action)

            if current_candidate is not None:
                if (current_candidate.score + self._threshold) >= best.score:
                    return current_candidate

        # 4. Переключение на новое действие (Исправление ошибки типа)
        if best.action_type != self._current_action:
            # Трюк для PyCharm: сохраняем в локальную переменную
            new_action_type: ActionType = best.action_type
            self._current_action = new_action_type
            self._hold_counter = self._min_hold

        return best

    def force_reset(self) -> None:
        """Reset hysteresis state (e.g. after a goal or set piece)."""
        self._current_action = None
        self._hold_counter = 0

    @staticmethod
    def _find_action(
            candidates: list[ActionSpec], action_type: ActionType
    ) -> ActionSpec | None:
        for spec in candidates:
            if spec.action_type == action_type:
                return spec
        return None