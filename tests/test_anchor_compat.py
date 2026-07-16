from __future__ import annotations

import numpy as np


def test_get_joint_limits_uses_current_hand_model(monkeypatch) -> None:
    from geort.anchor.compat import get_joint_limits

    class Hand:
        def get_joint_limit(self):
            return [0.0, -1.0], [1.0, 2.0]

    class Model:
        @staticmethod
        def build_from_config(config, render):
            assert config == {"name": "right"}
            assert render is False
            return Hand()

    monkeypatch.setattr("geort.env.hand.HandKinematicModel", Model)
    lower, upper = get_joint_limits({"name": "right"})

    assert np.array_equal(lower, np.array([0.0, -1.0]))
    assert np.array_equal(upper, np.array([1.0, 2.0]))
