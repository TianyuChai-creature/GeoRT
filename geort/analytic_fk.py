"""Analytical differentiable FK via pytorch_kinematics from URDF.

Provides a drop-in replacement for the per-finger neural FK (FKModel)
with exact kinematics.  Not yet wired into the training pipeline.
"""

from __future__ import annotations

from typing import Sequence

import pytorch_kinematics as pk
import torch
import torch.nn as nn


# Per-finger joint name blocks in the config joint_order (and URDF order).
# Order per finger: MCP2 (abduction α), MCP1 (flexion β1), PIP (β2), DIP (β3).
FINGER_JOINT_BLOCKS: tuple[tuple[str, ...], ...] = (
    ("F1-R-MCP2", "F1-R-MCP1", "F1-R-PIP", "F1-R-DIP"),
    ("F2-R-MCP2", "F2-R-MCP1", "F2-R-PIP", "F2-R-DIP"),
    ("F3-R-MCP2", "F3-R-MCP1", "F3-R-PIP", "F3-R-DIP"),
    ("F4-R-MCP2", "F4-R-MCP1", "F4-R-PIP", "F4-R-DIP"),
    ("F5-R-MCP2", "F5-R-MCP1", "F5-R-PIP", "F5-R-DIP"),
)

# Tip link names matching the finger order above.
TIP_LINK_NAMES: tuple[str, ...] = (
    "F1-R-DIP",
    "F2-R-DIP",
    "F3-R-DIP",
    "F4-R-DIP",
    "F5-R-DIP",
)


class AnalyticFK(nn.Module):
    """Exact differentiable forward kinematics from a hand URDF.

    Input:  normalised joint angles  [B, 20]  in [-1, 1] (config joint_order).
            Internally un-normalises to physical radians via
                physical = lower + (normalised + 1) * (upper - lower) / 2.
            The round-trip through float32 introduces ~1 μm tip-position noise;
            keep motion_delta >= 0.002 (~0.1 mm) to stay above the noise floor
            (see test_analytic_fk.py noise-floor calibration).

    Output: tip positions             [B,  5, 3] in metres (base_link frame).

    Supports batch, autograd, and follows the input device.
    """

    def __init__(
        self,
        urdf_path: str,
        joint_lower: Sequence[float],
        joint_upper: Sequence[float],
        tip_offsets: Sequence[Sequence[float]] | None = None,
    ) -> None:
        """
        Args:
            urdf_path: Path to the hand URDF file.
            joint_lower: Lower joint limits (rad) in config joint_order [20].
            joint_upper: Upper joint limits (rad) in config joint_order [20].
            tip_offsets:  Tip centre offsets in the distal link local frame,
                one [x, y, z] per finger (default: zeros).
        """
        super().__init__()
        with open(urdf_path) as fh:
            urdf_text = fh.read()

        self._chain = pk.build_chain_from_urdf(urdf_text)
        self._tip_links = list(TIP_LINK_NAMES)

        # Flatten the per-finger joint blocks into the 20-DOF ordered list.
        self._joint_names: list[str] = []
        self._per_finger_indices: list[list[int]] = []
        offset = 0
        for block in FINGER_JOINT_BLOCKS:
            self._joint_names.extend(block)
            self._per_finger_indices.append(list(range(offset, offset + len(block))))
            offset += len(block)

        if len(self._joint_names) != 20:
            raise AssertionError(
                f"Expected 20 joints from FINGER_JOINT_BLOCKS, got {len(self._joint_names)}"
            )

        # Verify that the URDF chain contains every joint we expect and that
        # the URDF order matches our config order (block-by-block).
        chain_names = self._chain.get_joint_parameter_names()
        # The chain may contain extra fixed (non-parameter) joints.  Find the
        # indices of our 20 parameterised joints in the chain order.
        self._chain_idx: list[int] = []
        chain_param_idx = 0
        for name in chain_names:
            if name in self._joint_names:
                expected = self._joint_names[len(self._chain_idx)]
                if name != expected:
                    raise AssertionError(
                        f"URDF joint order mismatch at position {len(self._chain_idx)}: "
                        f"expected {expected!r}, got {name!r}. "
                        f"Full chain order: {chain_names}"
                    )
                self._chain_idx.append(chain_param_idx)
            chain_param_idx += 1

        if len(self._chain_idx) != 20:
            missing = set(self._joint_names) - set(
                chain_names[i] for i in self._chain_idx if i < len(chain_names)
            )
            raise ValueError(
                f"Could not locate all 20 finger joints in the URDF chain. "
                f"Missing: {sorted(missing)}.  Chain names: {chain_names}"
            )

        # Normalisation parameters (same as HandFormatter).
        self.register_buffer(
            "_lower", torch.tensor(joint_lower, dtype=torch.float32)
        )
        self.register_buffer(
            "_upper", torch.tensor(joint_upper, dtype=torch.float32)
        )

        # Verify 4-DOF per-finger contract.
        for fi, indices in enumerate(self._per_finger_indices):
            names = [self._joint_names[i] for i in indices]
            if len(names) != 4:
                raise AssertionError(
                    f"Finger {fi} has {len(names)} DOF, expected 4: {names}"
                )
            expected_suffixes = ("MCP2", "MCP1", "PIP", "DIP")
            for name, suffix in zip(names, expected_suffixes):
                if not name.endswith(suffix):
                    raise AssertionError(
                        f"Finger {fi} joint {name!r}: expected suffix {suffix!r}. "
                        f"4-DOF order must be MCP2, MCP1, PIP, DIP."
                    )

        # Tip centre offsets in distal link local frame (matches SAPIEN convention).
        if tip_offsets is None:
            tip_offsets = [[0.0, 0.0, 0.0]] * 5
        if len(tip_offsets) != 5:
            raise ValueError(f"Expected 5 tip offsets, got {len(tip_offsets)}")
        self.register_buffer(
            "_tip_offsets",
            torch.tensor(tip_offsets, dtype=torch.float32).view(5, 3, 1),
        )

    def forward(
        self,
        joint_normalized: torch.Tensor,
        *,
        return_link_rotations: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute tip positions from normalized joint angles.

        Args:
            joint_normalized: [B, 20] float tensor in [-1, 1],
                ordered as FINGER_JOINT_BLOCKS (config joint_order).
            return_link_rotations: Also return distal-link rotations in the
                base frame. The default preserves the established position-only
                forward path exactly.

        Returns:
            Normally [B, 5, 3] positions in metres. When
            ``return_link_rotations`` is true, returns ``(tips, rotations)``
            with rotations shaped [B, 5, 3, 3].
        """
        if joint_normalized.ndim != 2 or joint_normalized.shape[1] != 20:
            raise ValueError(
                f"Expected joint_normalized [B, 20], got {tuple(joint_normalized.shape)}"
            )

        device = joint_normalized.device
        self.to(device)

        # Unnormalise: [-1, 1] → physical radians.
        half_range = (self._upper - self._lower) / 2.0
        physical = self._lower + (joint_normalized + 1.0) * half_range

        # Build the joint dict for pytorch_kinematics.
        # All non-finger joints (WRIST-PALM-R, PALM-R) are set to zero.
        B = physical.shape[0]
        chain_param_names = self._chain.get_joint_parameter_names()
        th: dict[str, torch.Tensor] = {}
        param_pos = 0
        for name in chain_param_names:
            if param_pos in self._chain_idx:
                # Map back to our 20-DOF order.
                our_idx = self._chain_idx.index(param_pos)
                th[name] = physical[:, our_idx]
            else:
                # Fixed joint — must match batch size.
                th[name] = torch.zeros(B, device=device)
            param_pos += 1

        # Forward kinematics.  Keep the chain on the same device as input.
        ret = self._chain.to(device=device).forward_kinematics(th)

        # Extract tip positions with centre offsets applied in link-local frame.
        tips = []
        link_rotations = []
        for i, link in enumerate(self._tip_links):
            m = ret[link].get_matrix()  # [B, 4, 4]
            # Apply offset in the link's local frame (rotation only, then add translation).
            link_pos = m[:, :3, 3]  # [B, 3]
            link_rot = m[:, :3, :3]  # [B, 3, 3]
            offset_world = (link_rot @ self._tip_offsets[i]).squeeze(-1)  # [B, 3]
            tips.append(link_pos + offset_world)
            if return_link_rotations:
                link_rotations.append(link_rot)

        stacked_tips = torch.stack(tips, dim=1).to(device)  # [B, 5, 3]
        if return_link_rotations:
            return stacked_tips, torch.stack(link_rotations, dim=1).to(device)
        return stacked_tips

    @property
    def n_dof(self) -> int:
        return 20

    @property
    def n_keypoints(self) -> int:
        return 5
