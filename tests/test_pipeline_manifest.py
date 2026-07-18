from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from geort.anchor import parity
from geort.anchor.compat import get_joint_limits
from geort.analytic_fk import AnalyticFK
from geort.pipeline.manifest import load_hand_manifest
from geort.utils.config_utils import get_config, parse_config_keypoint_info, select_keypoint_types
import torch


def _write_manifest(path: Path, *, output_root: str = "outputs/demo") -> Path:
    path.write_text(
        "hand_id: demo\n"
        "urdf: assets/custom_left/URDF_L.urdf\n"
        "hts: data/hts_left.npy\n"
        "hand_config: custom_left\n"
        "side: L\n"
        f"output_root: {output_root}\n",
        encoding="utf-8",
    )
    return path


def test_load_hand_manifest_resolves_paths_and_current_joint_limits(tmp_path: Path) -> None:
    manifest = load_hand_manifest(_write_manifest(tmp_path / "hand.yaml"))

    assert manifest.hand_id == "demo"
    assert manifest.side == "L"
    assert manifest.output_root == Path("outputs/demo")
    assert manifest.urdf.is_file()
    assert manifest.hts.is_file()
    lower, upper = manifest.joint_limits()
    assert lower.shape == upper.shape == (20,)



def test_analytic_fk_uses_explicit_left_side_token() -> None:
    config = get_config("custom_left")
    lower, upper = get_joint_limits(config)
    tips = select_keypoint_types(parse_config_keypoint_info(config), allowed_types=("tip",))

    fk = AnalyticFK(config["urdf_path"], lower, upper, tip_offsets=tips["offset"], side="L")

    assert fk(torch.zeros((2, 20), dtype=torch.float32)).shape == (2, 5, 3)


def test_load_hand_manifest_rejects_noncanonical_output_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="output_root"):
        load_hand_manifest(_write_manifest(tmp_path / "hand.yaml", output_root="outputs/not-demo"))


def test_manifest_parity_writes_output_and_docs_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _write_manifest(tmp_path / "hand.yaml")
    gates_path = tmp_path / "gates.yaml"
    gates_path.write_text("fk_parity_max_m: 0.000001\n", encoding="utf-8")

    def fake_write(path: Path, config: dict) -> Path:
        path.write_bytes(b"qpos")
        return path

    monkeypatch.setattr(parity, "write_parity_qpos", fake_write)
    monkeypatch.setattr(
        parity,
        "compare_parity_qpos",
        lambda path, config, **_: {
            "overall": {"max_m": 1.0e-7, "mean_m": 1.0e-8},
            "fingers": {},
        },
    )

    output_root = Path.cwd() / "outputs/demo"
    try:
        report = parity.main(
            [
                "--manifest", str(manifest_path),
                "--gates", str(gates_path),
                "--docs-root", str(tmp_path / "docs/reports"),
            ]
        )

        assert (output_root / "parity/parity_qpos.npz").exists()
        assert (output_root / "parity/parity_report.json").exists()
        assert (tmp_path / "docs/reports/demo_fk_parity_m1.md").exists()
        assert report["gate"]["passed"] is True
    finally:
        shutil.rmtree(output_root, ignore_errors=True)
