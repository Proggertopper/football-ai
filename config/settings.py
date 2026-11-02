"""Typed configuration loader.

Reads YAML config and exposes it as nested dataclasses with attribute access.
Falls back to default.yaml shipped with the package when no path is given.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "default.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*, returning a new dict."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


class _Namespace:
    """Dict-like object with attribute access for nested config sections.

    Dicts whose keys are all strings become nested _Namespaces.
    Dicts with non-string keys (e.g. YAML integer keys like class_map)
    are stored as plain dicts to avoid setattr errors.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        for key, value in data.items():
            str_key = str(key)
            if isinstance(value, dict) and all(isinstance(k, str) for k in value):
                setattr(self, str_key, _Namespace(value))
            else:
                setattr(self, str_key, value)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, _Namespace):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"Namespace({items})"


@dataclass
class Config:
    """Top-level configuration container.

    Sections are accessible as attributes:
        cfg.vision.confidence_threshold
        cfg.fc26.green_timing_delay_ms
    """

    capture: _Namespace = field(default_factory=lambda: _Namespace({}))
    vision: _Namespace = field(default_factory=lambda: _Namespace({}))
    tracker: _Namespace = field(default_factory=lambda: _Namespace({}))
    world: _Namespace = field(default_factory=lambda: _Namespace({}))
    state: _Namespace = field(default_factory=lambda: _Namespace({}))
    decision: _Namespace = field(default_factory=lambda: _Namespace({}))
    actions: _Namespace = field(default_factory=lambda: _Namespace({}))
    fc26: _Namespace = field(default_factory=lambda: _Namespace({}))
    tactics: _Namespace = field(default_factory=lambda: _Namespace({}))
    observe: _Namespace = field(default_factory=lambda: _Namespace({}))
    performance: _Namespace = field(default_factory=lambda: _Namespace({}))

    _raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return self._raw


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Load configuration from YAML files.

    1. Always loads the built-in default.yaml first.
    2. If *path* is given, deep-merges the user file on top.
    3. If *overrides* dict is given, deep-merges that last.
    """
    with open(_DEFAULT_CONFIG_PATH, "r") as f:
        base: dict[str, Any] = yaml.safe_load(f)

    if path is not None:
        with open(path, "r") as f:
            user: dict[str, Any] = yaml.safe_load(f) or {}
        base = _deep_merge(base, user)

    if overrides:
        base = _deep_merge(base, overrides)

    cfg = Config(_raw=base)
    for section_name, section_data in base.items():
        if isinstance(section_data, dict):
            setattr(cfg, section_name, _Namespace(section_data))


    return cfg