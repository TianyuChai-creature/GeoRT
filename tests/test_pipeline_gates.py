from __future__ import annotations

from pathlib import Path

from geort.pipeline.gates import evaluate_fk_parity_gate, load_gate_spec


def test_fk_parity_gate_is_inclusive_at_limit(tmp_path: Path) -> None:
    path = tmp_path / "gates.yaml"
    path.write_text("fk_parity_max_m: 0.000001\n", encoding="utf-8")
    spec = load_gate_spec(path)

    assert evaluate_fk_parity_gate({"overall": {"max_m": 1.0e-6}}, spec)["passed"] is True
    assert evaluate_fk_parity_gate({"overall": {"max_m": 1.1e-6}}, spec)["passed"] is False
