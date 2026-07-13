# AnyDexRT Step6 Sparse Anchor Design

## Goal

Implement AnyDexRT few-shot human guidance for the current TIP-only
`FK(IK(x))` pipeline. The operator records 50 sparse, per-finger HTS poses.
The pipeline interpolates them into 750 paired human/robot fingertip anchors
and trains `L_align` only on the finger identified by each anchor.

## Paper Contract

The implementation follows AnyDexRT Appendix D:

- Two anchor types: lateral rotation and bending.
- `K0=5` sparse human records for each finger and anchor type.
- Lateral rotation fixes `beta1=beta2=beta3=0` and varies `alpha`.
- Bending fixes `alpha=0` and uses
  `beta1=beta2=lambda*beta3`, with `lambda=2`.
- The four non-thumb fingers sample `beta1` at
  `[0, pi/8, pi/4, 3*pi/8, pi/2]`.
- The thumb uses five samples from a feasible pre-generated URDF bending
  trajectory because its bending parameterization and limits differ.
- P2 linearly interpolates lateral rotation to `K=50` and bending to
  `K=100`.
- P3 generates robot anchors through URDF forward kinematics at matching
  trajectory parameters.

Reference: <https://arxiv.org/html/2607.08341>

## Sparse Collection Order

Collection contains exactly 50 records:

1. Finger order: thumb, index, middle, ring, pinky.
2. For each finger: lateral rotation levels 0 through 4.
3. For the same finger: bending levels 0 through 4.

Each record supervises only its target finger. Although the HTS receiver
provides all 21 landmarks, collection stores the complete averaged HTS frame
for diagnostics and records the target finger index explicitly. The final
training anchors extract only the target TIP.

## Robot Reference Poses

The four joints in every config finger group are interpreted in existing
joint order as:

1. `alpha`: MCP2 lateral rotation.
2. `beta1`: MCP1 flexion.
3. `beta2`: PIP flexion.
4. `beta3`: DIP flexion.

Lateral rotation uniformly samples the active MCP2 URDF limit from lower to
upper while setting the other three joints to zero clipped into their limits.

For index through pinky, bending sets
`[alpha, beta1, beta2, beta3] = [0, b, b, b/lambda]` for the five paper
values, then clips only for floating-point boundary tolerance. A material
limit violation is an error rather than a silent trajectory change.

Thumb bending uses a feasible trajectory from the neutral pose to its
configured flexion upper limits. The five sparse poses are uniformly spaced
on that trajectory. The exact qpos vectors are saved with collection metadata
so P1 display and P3 generation use identical references.

All non-target joints remain at their neutral value, defined as zero clipped
to the current URDF limits.

## Collection Interaction

`geort/anchor/collect_human_anchors.py` owns the interactive P1 workflow:

- Build a non-headless SAPIEN hand and start the selected HTS stream.
- Show the current robot reference qpos continuously.
- Print record number, target finger, anchor type, level, and a concrete
  operator instruction in the terminal.
- Wait for Enter while continuing to refresh the viewer.
- After Enter, collect a configurable 1.5 second window and average finite
  HTS frames.
- Require a configurable minimum number of frames; reject and retry a record
  if the stream is stale, too short, or non-finite.
- Save after every accepted record to a partial NPZ plus JSON metadata.
- Resume only when hand, side, spec version, and task order match exactly.
- Write the final sparse NPZ after all 50 records; never overwrite a complete
  collection unless the operator passes an explicit overwrite flag.

The collector prints instructions in Chinese. Slight imitation offsets are
accepted, consistent with the paper.

## Data Contracts

### Sparse P1 output

`data/human_anchors_custom_right_sparse.npz` for `custom_right`, or
`data/human_anchors_custom_left_sparse.npz` for `custom_left`:

- `human_frames`: `[50, 21, 3]` averaged raw HTS frames.
- `robot_qpos`: `[50, D]` displayed reference qpos.
- `finger_indices`: `[50]`, values 0 through 4 in TIP order.
- `anchor_types`: `[50]`, strings `lateral` or `bending`.
- `levels`: `[50]`, values 0 through 4.
- `trajectory_t`: `[50]`, normalized values 0 through 1.
- `frame_counts`: `[50]`, accepted HTS frames per window.

### Final P2/P3 output

`data/anchors_custom_right.npz` for `custom_right`, or
`data/anchors_custom_left.npz` for `custom_left`:

- `human_points`: `[750, 3]`, interpolated target human TIPs.
- `robot_points`: `[750, 3]`, matching exact-URDF target TIPs.
- `finger_indices`: `[750]`.
- `anchor_types`: `[750]`.
- `trajectory_t`: `[750]`.
- `source_sparse_indices`: `[750, 2]`, neighboring P1 records.

Counts are `5*50=250` lateral anchors plus `5*100=500` bending anchors.
Interpolation includes both sparse endpoints and preserves exact endpoint
values.

The prepared manifest `anchors` object records the NPZ filename,
`normalized: false`, schema version, sparse source, and counts.

## Trainer Integration

The prepared-data loader accepts sparse anchor arrays plus
`finger_indices`. It normalizes each human/robot point using the center and
scale for that row's finger.

For an anchor batch:

1. Create a zero TIP tensor of shape `[B, 5, 3]`.
2. Scatter each normalized human anchor into its row's target finger.
3. Run the existing independent per-finger IK networks and frozen FK.
4. Gather only the mapped target robot TIP.
5. Compute `anchor_align_loss(gathered_mapped, robot_anchor)`.

Zero placeholders do not affect the target output because every IK finger MLP
reads only its own TIP. Non-target mapped outputs are never included in
`L_align`. The Step5 P-Chamfer, distance, and motion objectives remain
unchanged and equally weighted; `L_align` is added with weight 1 when
anchors are present.

## Generation and Visualization

`geort/anchor/interpolate.py` performs deterministic piecewise-linear
interpolation and exposes pure functions for testing.

`geort/anchor/generate_robot_anchors.py` validates the sparse contract,
generates the 750 paired anchors, updates the selected prepared manifest
atomically, and prints per-finger/type counts.

The generator also writes a lightweight Plotly HTML report containing one
3D trace pair per finger and anchor type. Human and robot trajectories use
separate panels because their normalization is per-domain; the report is for
trajectory direction and ordering, not metric-space overlay.

## Failure Handling

- Reject malformed HTS frames, missing fingers, unexpected task order,
  non-finite values, duplicate final output, and manifest/config mismatches.
- Never silently reuse anchors from another hand or URDF config.
- Refuse interpolation unless every finger/type group has exactly five
  ordered sparse records.
- Write manifest changes through a temporary file followed by replacement.
- Runtime artifacts remain ignored by Git.

## Verification

Automated tests cover:

- Exactly 50 sparse tasks in the defined order.
- Paper equations and configured joint-limit validation.
- Thumb trajectory feasibility.
- Interpolation to 50/100 with exact endpoints.
- Final 750-row ordering and finger indices.
- P1 stable-window validation and resumable save contract without HTS/SAPIEN.
- P3 exact FK target selection and manifest update.
- Per-finger normalization and sparse gather behavior in trainer.
- `L_align` affects only the target finger.

Manual H3 then runs the collector for the right hand. After generation and
training, visual acceptance compares trajectory trends and replay behavior
against the Step5 checkpoint.
