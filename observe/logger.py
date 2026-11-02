"""Structured event logger.

Logs every decision, state snapshot, and action to a JSONL file
with timestamps. Designed for post-match analysis and debugging.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from config.settings import Config

logger = logging.getLogger(__name__)


class EventLogger:
    """Append-only JSONL event logger.

    Each line is a JSON object with:
        {"ts": <epoch_ms>, "frame": <int>, "type": <str>, "data": {...}}
    """

    def __init__(self, cfg: Config) -> None:
        log_path = Path(getattr(cfg.observe, 'log_file', 'logs/events.jsonl'))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = log_path
        self._level = getattr(cfg.observe, 'log_level', 'INFO')
        self._file = open(log_path, "a", encoding="utf-8")
        self._frame: int = 0

    def set_frame(self, frame_id: int) -> None:
        self._frame = frame_id

    def log(self, event_type: str, data: dict[str, Any]) -> None:
        """Write a structured event."""
        record = {
            "ts": time.time() * 1000,
            "frame": self._frame,
            "type": event_type,
            "data": data,
        }
        line = json.dumps(record, default=str)
        self._file.write(line + "\n")

    def log_decision(
            self,
            chosen_action: str,
            score: float,
            risk: float,
            ev: float,
            metadata: dict[str, Any] | None = None,
    ) -> None:
        self.log("decision", {
            "action": chosen_action,
            "score": round(score, 4),
            "risk": round(risk, 4),
            "expected_value": round(ev, 4),
            "metadata": metadata or {},
        })

    def log_state(self, state_summary: dict[str, Any]) -> None:
        self.log("state", state_summary)

    def log_action_executed(self, action: str, duration_frames: int) -> None:
        self.log("action_exec", {"action": action, "duration": duration_frames})

    def log_tactical_mode(self, mode: str, frames_in_mode: int) -> None:
        self.log("tactical_mode", {"mode": mode, "frames": frames_in_mode})

    def log_performance(self, metrics: dict[str, float]) -> None:
        self.log("performance", metrics)

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        self._file.flush()
        self._file.close()