from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def test_step1_legacy_modules_are_removed() -> None:
    removed = [
        "geort/mocap/search_custom_aa_limits.py",
        "geort/mocap/hts_balance.py",
        "geort/mocap/hts_stage3.py",
        "geort/mocap/hts_prepare_training.py",
        "geort/training_targets.py",
        "geort/dataset_manifest.py",
    ]
    assert [path for path in removed if (ROOT / path).exists()] == []


def test_step1_legacy_mechanisms_are_absent_from_geort() -> None:
    pattern = re.compile(
        r"aa_limit|mold|chamfer_target|fist|pinch|segment_direction|"
        r"loss_weight|weights_path|WeightedRandomSampler|hts_balance|"
        r"hts_stage3|dataset_manifest",
        re.IGNORECASE,
    )
    matches = []
    for path in sorted((ROOT / "geort").rglob("*")):
        if path.suffix not in {".py", ".json"}:
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if pattern.search(line):
                matches.append(
                    f"{path.relative_to(ROOT)}:{line_number}:{line.strip()}"
                )
    assert matches == []
