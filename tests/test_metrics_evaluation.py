from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "geort" / "mocap" / "metrics_evaluation.py"
spec = importlib.util.spec_from_file_location("metrics_evaluation", MODULE_PATH)
assert spec is not None and spec.loader is not None
metrics = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = metrics
spec.loader.exec_module(metrics)


def test_compute_rest_offset_reports_fraction_of_joint_range() -> None:
    qpos = np.array([[0.0, 0.5], [0.2, 0.7]], dtype=np.float32)
    q_default = np.array([0.0, 0.5], dtype=np.float32)
    q_low = np.array([-1.0, 0.0], dtype=np.float32)
    q_high = np.array([1.0, 1.0], dtype=np.float32)

    result = metrics.compute_rest_offset(qpos, q_default=q_default, q_low=q_low, q_high=q_high)

    assert result["per_joint_median"] == pytest.approx([0.05, 0.1])
    assert result["max_median"] == pytest.approx(0.1)
    assert result["passes_5_percent"] is False


def test_compute_saturation_rate_splits_lower_and_upper_sides() -> None:
    qpos = np.array([[-0.96, 0.0], [0.0, 0.96], [0.5, 0.5]], dtype=np.float32)
    q_low = np.array([-1.0, -1.0], dtype=np.float32)
    q_high = np.array([1.0, 1.0], dtype=np.float32)

    result = metrics.compute_saturation_rate(qpos, q_low=q_low, q_high=q_high, margin=0.05)

    assert result["lower_per_joint"] == pytest.approx([1 / 3, 0.0])
    assert result["upper_per_joint"] == pytest.approx([0.0, 1 / 3])
    assert result["any_lower"] == pytest.approx(1 / 3)
    assert result["any_upper"] == pytest.approx(1 / 3)


def test_baseline_gate_fails_when_uniform_has_no_pathology() -> None:
    gate = metrics.evaluate_baseline_gate(
        uniform_metrics={
            "signed_gain": {"median": 1.05},
            "saturation_rate": {"any_lower": 0.0, "any_upper": 0.0},
            "rest_offset": {"max_median": 0.03},
        }
    )

    assert gate["status"] == "failed_baseline_not_pathological"


def test_baseline_gate_passes_when_any_pathology_is_present() -> None:
    gate = metrics.evaluate_baseline_gate(
        uniform_metrics={
            "signed_gain": {"median": 1.8},
            "saturation_rate": {"any_lower": 0.0, "any_upper": 0.0},
            "rest_offset": {"max_median": 0.03},
        }
    )

    assert gate["status"] == "passed"
