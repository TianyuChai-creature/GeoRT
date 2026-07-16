from __future__ import annotations

import numpy as np
import torch


def test_analytic_tip_callback_accepts_physical_qpos(monkeypatch) -> None:
    from geort.anchor.compat import make_analytic_tip_callback

    class FK:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, normalized):
            return torch.stack((normalized[:, :3],) * 5, dim=1)

    monkeypatch.setattr("geort.analytic_fk.AnalyticFK", FK)
    callback = make_analytic_tip_callback(
        {"urdf_path": "ignored"}, np.zeros(3), np.ones(3), [[0, 0, 0]] * 5
    )

    assert np.allclose(callback(np.array([0.5, 0.5, 0.5]), 3), [0.0, 0.0, 0.0])
