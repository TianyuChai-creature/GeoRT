"""CLI contract for the C2eL canonical evaluator."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_c2el_s42.py"


def test_canonical_evaluator_accepts_explicit_anchor_bundle() -> None:
    """Anchor residuals must be evaluated against the selected bundle."""
    env = os.environ | {"PYTHONPATH": str(ROOT)}
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--anchor-path" in result.stdout
