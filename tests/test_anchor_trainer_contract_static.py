"""Static guardrails for sparse-anchor trainer integration."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_trainer_exposes_anchor_cli_and_contract_wiring() -> None:
    source = (ROOT / "geort" / "trainer.py").read_text(encoding="utf-8")

    assert 'parser.add_argument("--anchor_path"' in source
    assert 'parser.add_argument("--w_anchor", type=float, default=1.0' in source
    assert "load_raw_anchor_training_points" in source
    assert "归一化契约尚未写入" in source
    assert "human_data_source mismatch" in source
    assert "anchor_loss" in source
    assert "anchor_batch_size = 32" in source


def test_trainer_exposes_explicit_formal_run_batch_and_lr_controls() -> None:
    source = (ROOT / "geort" / "trainer.py").read_text(encoding="utf-8")

    assert 'parser.add_argument("--batch_size", type=int, default=2048' in source
    assert 'parser.add_argument("--lr", type=float, default=1e-4' in source
    assert "batch_size=args.batch_size" in source
    assert "lr=args.lr" in source
    assert '"batch_size": batch_size' in source
    assert '"lr": lr' in source


def test_trainer_logs_a_structured_startup_configuration() -> None:
    source = (ROOT / "geort" / "trainer.py").read_text(encoding="utf-8")

    assert 'print("trainer config:"' in source
    assert '"batch_size": batch_size' in source
    assert '"anchor_path": str(anchor_path) if anchor_path else None' in source
