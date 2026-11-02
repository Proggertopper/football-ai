"""Replay capture — save frame-by-frame state and decisions.

Supports JSON and CSV formats. Replay files can be loaded by the
evaluation harness for regression testing and analysis.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from config.settings import Config


class ReplayRecorder:
    """Records per-frame snapshots to disk."""

    def __init__(self, cfg: Config, session_id: str = "default") -> None:
        self._enabled: bool = cfg.observe.replay_enabled
        self._format: str = cfg.observe.replay_format
        replay_dir = Path(cfg.observe.replay_dir)
        replay_dir.mkdir(parents=True, exist_ok=True)

        ext = "jsonl" if self._format == "json" else "csv"
        self._path = replay_dir / f"replay_{session_id}.{ext}"
        self._file = open(self._path, "w", encoding="utf-8", newline="") if self._enabled else None
        self._csv_writer: csv.DictWriter | None = None
        self._headers_written = False
        self._frame_count = 0

    def record_frame(self, data: dict[str, Any]) -> None:
        """Record a single frame's data."""
        if not self._enabled or self._file is None:
            return

        self._frame_count += 1

        if self._format == "json":
            self._file.write(json.dumps(data, default=str) + "\n")
        else:
            # CSV: flatten nested dicts
            flat = self._flatten(data)
            if not self._headers_written:
                self._csv_writer = csv.DictWriter(self._file, fieldnames=sorted(flat.keys()))
                self._csv_writer.writeheader()
                self._headers_written = True
            if self._csv_writer is not None:
                self._csv_writer.writerow(flat)

    def flush(self) -> None:
        if self._file is not None:
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def file_path(self) -> Path:
        return self._path

    @staticmethod
    def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        """Flatten nested dict with dot-separated keys."""
        items: dict[str, Any] = {}
        for k, v in d.items():
            key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
            if isinstance(v, dict):
                items.update(ReplayRecorder._flatten(v, key))
            else:
                items[key] = v
        return items