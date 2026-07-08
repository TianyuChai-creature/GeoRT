from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


training_targets = load_module("training_targets", "geort/training_targets.py")
hash_utils = load_module("hash_utils", "geort/utils/hash_utils.py")


def test_sha256_file_changes_with_content(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("one", encoding="utf-8")
    first = hash_utils.sha256_file(path)
    path.write_text("two", encoding="utf-8")
    second = hash_utils.sha256_file(path)

    assert first != second
    assert len(first) == 64
    assert len(second) == 64


def test_resolve_uniform_target_keeps_default_generation_path() -> None:
    resolved = training_targets.resolve_chamfer_target_path(
        hand_name="custom_right",
        chamfer_target="uniform",
        explicit_path=None,
    )

    assert resolved.path == Path("data/custom_right.npz")
    assert resolved.requires_existing is False


def test_resolve_human_target_requires_existing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="human-shaped chamfer target"):
        training_targets.resolve_chamfer_target_path(
            hand_name="custom_right",
            chamfer_target="human",
            explicit_path=tmp_path / "missing.npz",
        )


def test_build_training_metadata_hashes_target_and_mold(tmp_path: Path) -> None:
    target = tmp_path / "target.npz"
    mold = tmp_path / "mold.json"
    target.write_bytes(b"target")
    mold.write_text("{}", encoding="utf-8")

    metadata = training_targets.build_training_metadata(
        chamfer_target="human",
        target_path=target,
        mold_path=mold,
        human_data_path=tmp_path / "human.npy",
        n_epoch=200,
        loss_weights={"w_chamfer": 80.0},
        cli_args={"chamfer_target": "human"},
    )

    assert metadata["chamfer_target"] == "human"
    assert metadata["target_cloud"]["path"] == target.as_posix()
    assert metadata["target_cloud"]["sha256"] == hash_utils.sha256_file(target)
    assert metadata["mold"]["sha256"] == hash_utils.sha256_file(mold)
    assert metadata["n_epoch"] == 200


def test_save_training_metadata_writes_json(tmp_path: Path) -> None:
    path = tmp_path / "training_metadata.json"
    training_targets.save_training_metadata(path, {"a": 1})

    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}
