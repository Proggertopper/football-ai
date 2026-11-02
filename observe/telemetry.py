"""Telemetry aggregation pipeline.

Periodically flushes aggregated stats to the logger. Separates
high-frequency per-frame data from lower-frequency summary data.
"""

from __future__ import annotations

from config.settings import Config
from observe.logger import EventLogger
from observe.metrics import PerformanceMetrics


class TelemetryPipeline:
    """Aggregates metrics and flushes summaries at configured intervals."""

    def __init__(self, cfg: Config, logger: EventLogger, metrics: PerformanceMetrics) -> None:
        self._flush_interval: int = getattr(cfg.observe, 'telemetry_flush_interval', 100)
        self._logger = logger
        self._metrics = metrics
        self._frame_counter: int = 0
        self._decision_counts: dict[str, int] = {}
        self._mode_counts: dict[str, int] = {}

    def tick(self, frame_id: int) -> None:
        """Called every frame. Flushes when interval is reached."""
        self._frame_counter += 1
        if self._frame_counter >= self._flush_interval:
            self._flush(frame_id)
            self._frame_counter = 0

    def record_decision(self, action_name: str) -> None:
        self._decision_counts[action_name] = self._decision_counts.get(action_name, 0) + 1

    def record_mode(self, mode_name: str) -> None:
        self._mode_counts[mode_name] = self._mode_counts.get(mode_name, 0) + 1

    def _flush(self, frame_id: int) -> None:
        summary: dict = self._metrics.summary()
        summary.update({
            "decision_distribution": dict(self._decision_counts),
            "mode_distribution": dict(self._mode_counts)
        })
        self._logger.set_frame(frame_id)
        self._logger.log_performance(summary)
        self._logger.flush()
        self._decision_counts.clear()
        self._mode_counts.clear()

    def final_flush(self, frame_id: int) -> None:
        """Force a final flush at shutdown."""
        self._flush(frame_id)