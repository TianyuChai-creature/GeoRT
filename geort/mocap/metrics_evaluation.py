"""A/B acceptance metrics for GeoRT v3 distribution alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _as_float_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=np.float32).tolist()]


def compute_rest_offset(
    qpos: np.ndarray,
    *,
    q_default: np.ndarray,
    q_low: np.ndarray,
    q_high: np.ndarray,
) -> dict[str, Any]:
    qpos = np.asarray(qpos, dtype=np.float32)
    q_default = np.asarray(q_default, dtype=np.float32)
    q_low = np.asarray(q_low, dtype=np.float32)
    q_high = np.asarray(q_high, dtype=np.float32)
    span = np.maximum(q_high - q_low, 1e-8)
    offset = np.abs(qpos - q_default.reshape(1, -1)) / span.reshape(1, -1)
    per_joint_median = np.median(offset, axis=0).astype(np.float32)
    max_median = float(per_joint_median.max()) if per_joint_median.size else 0.0
    return {
        "per_joint_median": _as_float_list(per_joint_median),
        "max_median": max_median,
        "passes_5_percent": bool(max_median < 0.05),
    }


def compute_saturation_rate(
    qpos: np.ndarray,
    *,
    q_low: np.ndarray,
    q_high: np.ndarray,
    margin: float = 0.05,
) -> dict[str, Any]:
    qpos = np.asarray(qpos, dtype=np.float32)
    q_low = np.asarray(q_low, dtype=np.float32)
    q_high = np.asarray(q_high, dtype=np.float32)
    span = np.maximum(q_high - q_low, 1e-8)
    lower = qpos <= (q_low.reshape(1, -1) + margin * span.reshape(1, -1))
    upper = qpos >= (q_high.reshape(1, -1) - margin * span.reshape(1, -1))
    lower_per_joint = lower.mean(axis=0).astype(np.float32)
    upper_per_joint = upper.mean(axis=0).astype(np.float32)
    return {
        "lower_per_joint": _as_float_list(lower_per_joint),
        "upper_per_joint": _as_float_list(upper_per_joint),
        "any_lower": float(lower.any(axis=1).mean()) if lower.size else 0.0,
        "any_upper": float(upper.any(axis=1).mean()) if upper.size else 0.0,
    }


def evaluate_baseline_gate(uniform_metrics: dict[str, Any]) -> dict[str, Any]:
    gain_median = float(uniform_metrics.get("signed_gain", {}).get("median", 0.0))
    saturation = uniform_metrics.get("saturation_rate", {})
    rest_offset = uniform_metrics.get("rest_offset", {})
    has_gain_pathology = gain_median > 1.5
    has_saturation_pathology = (
        float(saturation.get("any_lower", 0.0)) > 0.0
        or float(saturation.get("any_upper", 0.0)) > 0.0
    )
    has_rest_pathology = float(rest_offset.get("max_median", 0.0)) > 0.05
    if has_gain_pathology or has_saturation_pathology or has_rest_pathology:
        return {
            "status": "passed",
            "gain_pathology": has_gain_pathology,
            "saturation_pathology": has_saturation_pathology,
            "rest_pathology": has_rest_pathology,
        }
    return {
        "status": "failed_baseline_not_pathological",
        "gain_pathology": False,
        "saturation_pathology": False,
        "rest_pathology": False,
    }


def save_metrics_json(path: Path | str, metrics: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Output metrics JSON path.")
    return parser


def main() -> None:
    build_arg_parser().parse_args()
    raise SystemExit(
        "Metric primitives are implemented. Full checkpoint A/B execution will be wired after model-run helpers are selected."
    )


if __name__ == "__main__":
    main()
