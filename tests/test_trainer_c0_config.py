from __future__ import annotations

import argparse
from pathlib import Path


def test_yaml_defaults_allow_explicit_cli_override(tmp_path: Path) -> None:
    from geort.trainer_cli import apply_yaml_defaults

    config = tmp_path / "equiv.yaml"
    config.write_text("fk_backend: neural\nchamfer_mode: bidirectional\nw_dist: 0\n", encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--fk_backend", default="analytic")
    parser.add_argument("--chamfer_mode", default="partial")
    parser.add_argument("--w_distance", type=float, default=1.0)

    apply_yaml_defaults(parser, config)
    args = parser.parse_args(["--fk_backend", "analytic"])

    assert args.fk_backend == "analytic"
    assert args.chamfer_mode == "bidirectional"
    assert args.w_distance == 0.0


def test_yaml_defaults_reject_unknown_keys(tmp_path: Path) -> None:
    import pytest
    from geort.trainer_cli import apply_yaml_defaults

    config = tmp_path / "invalid.yaml"
    config.write_text("not_a_trainer_flag: 1\n", encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--known", default=0)

    with pytest.raises(ValueError, match="unknown keys"):
        apply_yaml_defaults(parser, config)


def test_chamfer_mode_selects_one_way_or_bidirectional_loss() -> None:
    import torch
    from geort.trainer import chamfer_loss_for_keypoint

    mapped = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    target = torch.tensor([[[0.0, 0.0, 0.0]]])

    partial = chamfer_loss_for_keypoint(mapped, target, mode="partial")
    bidirectional = chamfer_loss_for_keypoint(mapped, target, mode="bidirectional")

    assert partial.item() == 1.0
    assert bidirectional.item() == 2.0
