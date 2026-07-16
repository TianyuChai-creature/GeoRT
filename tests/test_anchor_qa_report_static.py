from pathlib import Path

from geort.anchor.qa_report_static import QAInputs, build_report_record


ROOT = Path(__file__).resolve().parents[1]


def test_persisted_anchor_evidence_builds_all_requested_sections():
    record = build_report_record(
        QAInputs(
            hand="custom_right",
            human_data=ROOT / "data/hts_right.npy",
            human_anchors=ROOT / "data/anchors_human_right.npz",
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

    assert len(record["decision"]["parameter_percentiles"]) == 10
    assert len(record["decision"]["span_ratios"]) == 10
    assert record["robot_and_pairing"]["parity_composition"]["total"] == 750
    assert record["robot_and_pairing"]["parity"]["overall"]["max_m"] < 1e-3

