"""Small, testable YAML-default adapter for the trainer CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


_ALIASES = {"w_dist": "w_distance"}


def resolved_config_json(values: dict[str, Any]) -> str:
    """Render the complete parsed CLI namespace deterministically for run evidence."""
    return json.dumps(values, sort_keys=True)


def apply_yaml_defaults(parser: argparse.ArgumentParser, path: str | Path) -> dict[str, Any]:
    """Apply a mapping of parser defaults; later explicit CLI flags still win."""
    source = Path(path)
    loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"trainer config must be a mapping: {source}")
    values = {_ALIASES.get(str(key), str(key)): value for key, value in loaded.items()}
    # YAML 1.1 treats unquoted "on"/"off" as booleans; the public C0
    # contract intentionally uses contact_refine: off, so restore its CLI enum.
    if isinstance(values.get("contact_refine"), bool):
        values["contact_refine"] = "on" if values["contact_refine"] else "off"
    destinations = {action.dest for action in parser._actions}
    unknown = sorted(set(values).difference(destinations))
    if unknown:
        raise ValueError(f"trainer config contains unknown keys: {unknown}")
    parser.set_defaults(**values)
    return values
