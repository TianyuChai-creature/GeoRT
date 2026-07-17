import hashlib
import json
from pathlib import Path

import pytest


def _write_fixture(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint" / "c2_s42"
    checkpoint.mkdir(parents=True)
    weights = checkpoint / "last.pth"
    weights.write_bytes(b"c2 weights")
    sha = hashlib.sha256(weights.read_bytes()).hexdigest()
    (checkpoint / "training_metadata.json").write_text(json.dumps({
        "cli_args": {"motion_frame": "global"},
        "anchor": {"enabled": True, "count": 750, "path": "data/anchors.npz"},
    }))
    archive = tmp_path / "outputs" / "final_matrix"
    archive.mkdir(parents=True)
    (archive / "checkpoint_hashes.json").write_text(json.dumps({
        "runs": {"c2_s42": {"checkpoint": "checkpoint/c2_s42", "last_pth_sha256": sha,
                              "motion_frame": "global",
                              "anchor": {"enabled": True, "count": 750, "path": "data/anchors.npz"}}}
    }))
    return checkpoint, archive


def test_verify_archived_checkpoint_returns_startup_summary(tmp_path):
    from geort.mocap.realtime_provenance import verify_archived_checkpoint

    checkpoint, archive = _write_fixture(tmp_path)
    result = verify_archived_checkpoint(checkpoint, archive, repo_root=tmp_path)

    assert result.run_id == "c2_s42"
    assert result.motion_frame == "global"
    assert result.anchor["count"] == 750


@pytest.mark.parametrize("mutation, expected", [
    ("weights", "SHA256"),
    ("motion", "motion_frame"),
    ("anchor", "anchor"),
])
def test_verify_archived_checkpoint_rejects_provenance_mismatch(tmp_path, mutation, expected):
    from geort.mocap.realtime_provenance import verify_archived_checkpoint

    checkpoint, archive = _write_fixture(tmp_path)
    if mutation == "weights":
        (checkpoint / "last.pth").write_bytes(b"different")
    else:
        metadata = json.loads((checkpoint / "training_metadata.json").read_text())
        if mutation == "motion":
            metadata["cli_args"]["motion_frame"] = "local"
        else:
            metadata["anchor"]["count"] = 32
        (checkpoint / "training_metadata.json").write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match=expected):
        verify_archived_checkpoint(checkpoint, archive, repo_root=tmp_path)


def test_build_ledger_registers_explicit_c2b_checkpoint(tmp_path):
    from geort.mocap.generate_realtime_checkpoint_ledger import build_ledger

    final_matrix = tmp_path / "final_matrix.json"
    final_matrix.write_text(json.dumps({"manifest": {"runs": {}}}))
    checkpoint = tmp_path / "checkpoint" / "c2b_s42"
    checkpoint.mkdir(parents=True)
    (checkpoint / "last.pth").write_bytes(b"c2b weights")
    (checkpoint / "training_metadata.json").write_text(json.dumps({
        "cli_args": {"motion_frame": "global"},
        "anchor": {"enabled": True, "count": 750, "path": "data/anchors.npz"},
    }))

    ledger = build_ledger(
        final_matrix,
        tmp_path,
        extra_checkpoints={"c2b_s42": checkpoint},
    )

    assert ledger["runs"]["c2b_s42"]["checkpoint"] == "checkpoint/c2b_s42"
    assert ledger["runs"]["c2b_s42"]["last_pth_sha256"] == hashlib.sha256(b"c2b weights").hexdigest()
