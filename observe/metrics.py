"""Performance metrics tracking.

Tracks FPS, latency per pipeline stage, detection confidence stats,
and exposes rolling averages for the debug overlay.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class StageTimer:
    """Accumulates timing samples for a named pipeline stage."""

    name: str
    _samples: deque[float] = field(default_factory=lambda: deque(maxlen=120))
    _start: float = 0.0

    def start(self) -> None:
        self._start = time.perf_counter()

    def stop(self) -> float:
        elapsed = (time.perf_counter() - self._start) * 1000  # ms
        self._samples.append(elapsed)
        return elapsed

    @property
    def avg_ms(self) -> float:
        if not self._samples:
            return 0.0
        return sum(self._samples) / len(self._samples)

    @property
    def max_ms(self) -> float:
        if not self._samples:
            return 0.0
        return max(self._samples)

    @property
    def last_ms(self) -> float:
        if not self._samples:
            return 0.0
        return self._samples[-1]


class PerformanceMetrics:
    """Central performance tracker for the pipeline.

    Usage:
        metrics.capture.start()
        frame = capture()
        metrics.capture.stop()
    """

    def __init__(self, window_size: int = 60) -> None:
        self.capture = StageTimer("capture")
        self.inference = StageTimer("inference")
        self.tracking = StageTimer("tracking")
        self.state_estimation = StageTimer("state_estimation")
        self.decision = StageTimer("decision")
        self.control = StageTimer("control")
        self.overlay = StageTimer("overlay")
        self.total = StageTimer("total")

        self._fps_timestamps: deque[float] = deque(maxlen=window_size)
        self._frame_count: int = 0
        self._detection_confidences: deque[float] = deque(maxlen=window_size)

    def tick_frame(self) -> None:
        """Call once per frame to update FPS tracking."""
        now = time.perf_counter()
        self._fps_timestamps.append(now)
        self._frame_count += 1

    @property
    def fps(self) -> float:
        if len(self._fps_timestamps) < 2:
            return 0.0
        elapsed = self._fps_timestamps[-1] - self._fps_timestamps[0]
        if elapsed < 1e-9:
            return 0.0
        return (len(self._fps_timestamps) - 1) / elapsed

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def record_detection_confidence(self, confidence: float) -> None:
        self._detection_confidences.append(confidence)

    @property
    def avg_detection_confidence(self) -> float:
        if not self._detection_confidences:
            return 0.0
        return sum(self._detection_confidences) / len(self._detection_confidences)

    def summary(self) -> dict[str, float]:
        """Return a dict of all metrics for logging/display."""
        return {
            "fps": round(self.fps, 1),
            "frame_count": self._frame_count,
            "capture_ms": round(self.capture.avg_ms, 2),
            "inference_ms": round(self.inference.avg_ms, 2),
            "tracking_ms": round(self.tracking.avg_ms, 2),
            "state_ms": round(self.state_estimation.avg_ms, 2),
            "decision_ms": round(self.decision.avg_ms, 2),
            "control_ms": round(self.control.avg_ms, 2),
            "overlay_ms": round(self.overlay.avg_ms, 2),
            "total_ms": round(self.total.avg_ms, 2),
            "avg_det_conf": round(self.avg_detection_confidence, 3),
        }