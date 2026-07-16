# L_motion Local Coordinate Frame Design

## Goal

Add an opt-in local-frame form of `L_motion` while preserving the current
global-frame training path byte-for-byte on CUDA.  The runtime controls are
`--motion_frame {global,local}` (default `global`) and `--device {cuda,cpu}`
(default `cuda`).

## Delivery Structure

### Commit A — device plumbing only

Replace trainer/FK/loader/loss device assumptions with one explicit
`torch.device`.  The default is CUDA and the commit contains no frame, FK
rotation, point-cloud schema, or motion-loss changes.  A deterministic,
explicit-parameter 50-step CUDA run is compared line-by-line and bit-for-bit
with the current `a4cdd46` CUDA baseline.

### Commit B — local frame implementation

Contains the implementation described below, tests, CPU-only diagnostic
scripts, and this specification.  A 50-step CUDA `--motion_frame global` run
is compared line-by-line and bit-for-bit with the Commit-A CUDA baseline.

## Runtime Contract

### Robot frames

`AnalyticFK.forward(q, return_link_rotations=True)` returns the existing TIP
positions plus the five distal-link rotations already present in the same
`forward_kinematics` result.  The no-keyword call returns only positions and
does not take a new FK path.

For each right-hand finger, the runtime task-frame orientation is

`R_task = R_link @ C_f`.

`C_f` is a checked-in numeric 3x3 constant.  It is derived once from the
configured TIP offset and the matching URDF DIP axis as follows:

`c1 = normalize(offset)`

`c2 = normalize(axis - dot(axis, c1) * c1)`

`c3 = cross(c1, c2)`

and `C_f = [c1 c2 c3]`.  If the physical DIP-positive sign check is negative,
`c2` and `c3` are flipped together before the literal is recorded.  `c1` is
never changed.  Source-derived constants are validated against the literals
at test time; there is no fitted or numeric calibration parameter.

The robot point-cloud generator writes a `keypoint_rotation` field aligned
with every point, and robot-anchor generation writes a `robot_rotation` field
aligned with every pair.  Existing bundles remain readable for global paths.
Any local-frame reader that needs a missing rotation field raises a named
schema error and never synthesizes an identity frame.

### Human frames

`geort.motion_frames` is the sole frame-construction module.  It accepts raw
HTS `[T,21,3]` metres and returns `[T,5,3,3]` matrices with column-vector
bases.

For non-thumb fingers, `e1 = normalize(TIP - DIP)` and the reference segment
is `PIP - DIP`.  For thumb, `e1 = normalize(TIP - IP)` and the reference
segment is `MCP - IP`.  The reference is projected off `e1`; its cross with
`e1` forms the provisional plane normal, and the third column completes a
right-handed basis.  The same DIP-positive sign convention as the robot
decides whether the second and third columns are flipped together.

Degeneracy is fixed before implementation: a segment norm below `1e-6 m` or
a post-projection relative sine below `1e-3` is degenerate.  A degenerate
finger frame reuses that finger's preceding valid frame.  A degenerate first
frame uses the identity matrix.  The cache records per-finger fallback counts
and source provenance.

When local mode is requested, frames are precomputed before the DataLoader is
created and written in the new run directory as `human_motion_frames.npz`.
`FramePointDataset` receives them as its `human_rotation` per-frame field.
Global mode does not build, read, or attach this field.

The human sign diagnostic uses the existing D1 `beta[..., finger, 2]` DIP
definition from `geort.anchor.human_geometry.estimate_finger_angles`; its
per-finger directional derivative is estimated from finite, palm-aligned D1
samples.  The reported scalar is
`dot(d e1 / d beta3, axis2 × e1)`.

### Loss and nearest-neighbour contract

`partial_chamfer_distance` receives an opt-in `return_indices` flag.  Its
normal call preserves its scalar result and call sequence.  In local mode it
returns the same scalar plus the already-computed P-Chamfer `argmin` indices.
The trainer gathers `keypoint_rotation` with those indices; it performs no
second `cdist` or nearest-neighbour operation.

The global branch retains the current exact call:

`local_motion_loss(d_human, d_robot)`.

The local branch supplies frames so the loss compares

`R_H(x)^T normalize(d_human)` and `R_R(f_m(x))^T normalize(d_robot)`.

The 32 anchor rows remain isolated in `L_align`; they never enter motion,
P-Chamfer, nullspace, or invalid-row statistics.

## Metadata and Evaluation

`motion_frame` and `device` are recorded in `training_metadata.json` and the
trainer startup configuration line.  Local-frame LMC evaluation imports the
human-frame helper from `geort.motion_frames`; it must not duplicate geometry.

CPU diagnostics report only numeric values for: orthogonality/determinant,
1000-qpos SAPIEN orientation parity, rigid-rotation invariance, NN distance
percentiles, fallback rate, raw offset-to-DIP-axis angles, and five human plus
five robot DIP-positive sign scalars.  CPU and CUDA outputs are never used for
bitwise comparisons.

## Non-goals

The default remains global.  No loss weight, nullspace setting, anchor data,
anchor path, URDF geometry, PIP offset data, site point, mining rule, or
local-vs-global A/B training is changed by this work.
