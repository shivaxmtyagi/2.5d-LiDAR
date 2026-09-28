"""Configuration loading. Every tunable parameter lives in configs/*.yaml —
nothing in the pipeline should hard-code thresholds directly."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """Thin dict subclass with attribute-style access and dotted-path get().

    Usage:
        cfg = Config.load("configs/default.yaml")
        cfg.sensor.max_range           # attribute style
        cfg.get("grid.breakpoints")    # dotted path
    """

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return _wrap(value)

    def get_path(self, dotted_path: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted_path.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return _wrap(node)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return _wrap(raw)


def _wrap(value: Any) -> Any:
    if isinstance(value, dict) and not isinstance(value, Config):
        return Config({k: _wrap(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value
