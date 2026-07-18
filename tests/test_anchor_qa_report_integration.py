from pathlib import Path

from geort.anchor.qa_report_runner import QAInputs, build_report_record
from tests.anchor_qa_fixture import write_human_anchor_bundle


ROOT = Path(__file__).resolve().parents[1]


def test_build_report_record_contains_requested_sections(tmp_path):
    human_anchors = write_human_anchor_bundle(tmp_path / "anchors_human_right.npz", ROOT / "data/hts_right.npy")
    record = build_report_record(
        QAInputs(
            hand="custom_right",
            human_data=ROOT / "data/hts_right.npy",
            human_anchors=human_anchors,
            parity_qpos=ROOT / "outputs/anchors/parity_qpos.npz",
            parity_report=ROOT / "outputs/anchors/custom_right_fk_parity.json",
            normalization_path=(
                ROOT
                / "checkpoint/custom_right_2026-07-16_10-08-30_seed42_null_v3_full"
                / "normalization.json"
            ),
            robot_data=ROOT / "data/custom_right.npz",
        )
    )

    assert set(record) >= {
        "decision",
        "human_self_check",
        "robot_and_pairing",
        "contract",
    }
    assert len(record["decision"]["parameter_percentiles"]) == 10
    assert len(record["decision"]["span_ratios"]) == 10
    assert record["robot_and_pairing"]["parity"]["overall"]["max_m"] < 1e-3

