"""Declarative acceptance gates shared by manifest-driven stations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class GateSpec:
    fk_parity_max_m: float


def load_gate_spec(path: Path | str) -> GateSpec:
    """Load the first hand-agnostic gate specification."""
    with Path(path).open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    if not isinstance(values, dict) or "fk_parity_max_m" not in values:
        raise ValueError("gate spec must define fk_parity_max_m")
    limit = float(values["fk_parity_max_m"])
    if not limit > 0.0:
        raise ValueError("fk_parity_max_m must be positive")
    return GateSpec(fk_parity_max_m=limit)


def evaluate_fk_parity_gate(report: Mapping[str, Any], spec: GateSpec) -> dict[str, Any]:
    """Return a serializable inclusive comparison for analytic/SAPIEN TIP parity."""
    observed = float(report["overall"]["max_m"])
    return {
        "name": "fk_parity_max",
        "observed_m": observed,
        "limit_m": spec.fk_parity_max_m,
        "passed": observed <= spec.fk_parity_max_m,
    }
