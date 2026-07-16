import torch
from pathlib import Path

import geort.trainer as trainer
from geort.analytic_fk import AnalyticFK
from geort.formatter import HandFormatter


def test_trainer_parser_defaults_to_cuda_and_accepts_cpu():
    parser = trainer.build_arg_parser()

    assert parser.parse_args([]).device == "cuda"
    assert parser.parse_args(["--device", "cpu"]).device == "cpu"


def test_analytic_fk_cpu_result_stays_on_input_device():
    lower = [-0.1] * 20
    upper = [0.1] * 20
    fk = AnalyticFK("assets/custom_right/URDF_R.urdf", lower, upper)
    qnorm = torch.zeros(2, 20)

    assert fk(qnorm).device.type == "cpu"


def test_trainer_has_no_direct_cuda_placement_in_training_path():
    source = Path("geort/trainer.py").read_text(encoding="utf-8")

    assert ".cuda()" not in source
    assert "device='cuda'" not in source
    assert "if args.device == 'cuda' and torch.cuda.is_available():" in source


def test_hand_formatter_uses_input_device():
    formatter = HandFormatter([-1.0, -1.0], [1.0, 1.0])
    values = torch.tensor([[0.0, 0.5]])

    assert formatter.normalize_torch(values).device == values.device
