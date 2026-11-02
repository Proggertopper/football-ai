from __future__ import annotations

from state.match_state import MatchState
from vision.geometry import Vec2

def predict_state(
        state: MatchState,
        horizon_sec: float,
        fps: float = 30.0,  # Это значение по умолчанию, если другое не передано
) -> tuple[Vec2, Vec2]:
    """
    Предсказывает позиции мяча и игрока на horizon_sec вперед.
    """
    # Вычисляем, сколько кадров пройдет за указанное время
    frames_ahead = horizon_sec * fps

    # Прогноз для мяча
    if state.ball is not None:
        # Используем встроенный метод объекта для расчета позиции через N кадров
        pred_ball = state.ball.predict_position(int(frames_ahead))
    else:
        pred_ball = Vec2.zero()

    # Прогноз для управляемого игрока
    if state.controlled_player is not None:
        pred_ctrl = state.controlled_player.predict_position(int(frames_ahead))
    else:
        pred_ctrl = Vec2.zero()

    return pred_ball, pred_ctrl