# AnyDexRT Step 6 D1-Mined Anchor Design

## Goal

Build AnyDexRT D2 anchors automatically from an existing D1 free-motion HTS
recording. No live HTS or SAPIEN collection is part of Step 6. The pipeline
mines five real human poses per finger and motion type, pairs them by level
with five robot poses sampled across the robot's own feasible range,
interpolates the pairs, and trains the existing TIP-only mapper with sparse
finger-indexed alignment.

Step 7 contact auto-labeling is a separate subsystem and will receive its own
design and implementation plan after Step 6.

## Inputs and Coordinate Contract

The miner accepts one raw HTS NPY with finite shape `[T, 21, 3]`, such as
`data/hts_right.npy`, plus `--hand-side right|left`. Landmark topology is:

- wrist: 0;
- thumb: CMC/MCP/IP/TIP at 1/2/3/4;
- index: MCP/PIP/DIP/TIP at 5/6/7/8;
- middle: MCP/PIP/DIP/TIP at 9/10/11/12;
- ring: MCP/PIP/DIP/TIP at 13/14/15/16;
- pinky: MCP/PIP/DIP/TIP at 17/18/19/20.

The existing capture conversion already stores both hands in GeoRT's common
right-handed convention. For each frame, the miner still removes residual
rigid hand motion by constructing a palm frame:

1. origin at the wrist;
2. longitudinal axis from wrist to middle MCP;
3. lateral axis from pinky MCP toward index MCP, orthogonalized against the
   longitudinal axis;
4. palm normal from the cross product of the lateral and longitudinal axes;
5. scale from index-MCP to pinky-MCP distance.

Frames with degenerate axes or palm scale are invalid and excluded. Medoid
descriptors use palm-frame, palm-scale-normalized landmarks, but the selected
output is always the untouched real raw frame at its source index.

## Human Angle Estimation

For index through pinky, let `m` be the wrist-to-MCP metacarpal direction and
`s1/s2/s3` the MCP-to-PIP, PIP-to-DIP, and DIP-to-TIP segment directions.
The miner computes:

- `alpha`: signed in-palm azimuth from projected `m` to projected `s1`;
- `beta1`: signed elevation/flexion of `s1` from the palm plane relative
  to `m`;
- `beta2`: flexion angle between `s1` and `s2`;
- `beta3`: flexion angle between `s2` and `s3`.

Angles use clipped dot products, `atan2`, and explicit finite/segment-length
validation. Tests use synthetic articulated fingers with known angles and
rigidly transformed copies to prove palm-frame invariance.

Thumb lateral rotation uses the same signed palm-frame principle with its
CMC-to-MCP and MCP-to-IP segments. Thumb bending does not use a beta coupling
model.

## Candidate Mining

There are two anchor types for every finger.

### Lateral

Lateral candidates require a nearly extended finger:

```text
max(abs(beta1), abs(beta2), abs(beta3)) <= straight_tol
```

The level parameter is `alpha`. Its asymmetric observed range is preserved;
the implementation never mirrors or symmetrizes it.

### Bending

For index through pinky, bending candidates require:

```text
abs(alpha) <= alpha_zero_tol
max(abs(beta1 - beta2), abs(beta1 - 2 * beta3)) <= coupling_tol
```

The level parameter is `beta1`. This filter follows the paper's
`beta1 = beta2 = 2 * beta3` coupling but treats it as a configurable
observation tolerance rather than forcing raw human poses onto the equation.

Default tolerances are:

- `straight_tol_deg=15`;
- `alpha_zero_tol_deg=10`;
- `coupling_tol_deg=20`.

If a finger/type has fewer than five candidates, the relevant action
tolerances are retried with factors `[1.0, 1.5, 2.0, 3.0, 4.0]`. Finite
validation and robust endpoint clipping are never relaxed. The first factor
that yields at least five candidates is used and recorded. Exhausting the
schedule is a hard failure.

## Geometric Five-Level Selection

Levels are geometric range fractions, not empirical-CDF quantiles:

```text
LEVEL_FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]
```

For each non-thumb finger/type:

1. compute the candidate level parameter;
2. use configurable `endpoint_quantiles=[0.02, 0.98]` only to reject tracking
   outliers and define robust lower/upper endpoints;
3. place five target values uniformly in angle space between those endpoints;
4. form a target neighborhood with initial half-width
   `level_band_fraction=0.025` of the robust range;
5. expand that half-width by `[1, 2, 4, 8]` until at least
   `min_level_support=5` candidates are available;
6. retain at most `max_medoid_candidates=256` frames nearest the target angle
   so exact medoid computation remains bounded;
7. choose the candidate minimizing summed Euclidean distance between
   palm-aligned target-finger landmark descriptors.

The selected values must be monotonic by level. One source frame may not
represent two levels of the same finger/type; if neighborhoods overlap, the
next-best unused medoid is selected. Endpoint support counts and every
neighborhood expansion are reported.

This procedure intentionally ignores dwell-time density around neutral.
The 25/50/75 percent levels span angle range rather than frame counts.

## Thumb Bending Main Trajectory

Thumb bending uses its palm-aligned, palm-scale-normalized TIP trajectory:

1. reject finite/geometry failures;
2. remove point outliers using the same configurable 2/98 percent robust
   projection endpoints;
3. estimate the dominant one-dimensional trajectory by PCA ordering followed
   by deterministic local medoids in `thumb_manifold_bins=64` ordered bins;
4. connect populated bin medoids into a polyline and compute cumulative 3D
   arc length;
5. place five targets uniformly at arc fractions
   `[0, 0.25, 0.5, 0.75, 1]`;
6. select distinct real source frames through the same bounded local-medoid
   rule.

The report includes PCA explained variance, populated-bin count, the polyline,
arc targets, and selected frames. At least five valid populated bins and five
distinct selected frames are required.

## Human Anchor Output

`geort/anchor/mine_human_anchors.py` writes
`data/anchors_human_<side>.npz` atomically:

- `human_frames`: `[50, 21, 3]` real raw frames;
- `human_points`: `[50, 3]` target TIPs from those frames;
- `source_indices`: `[50]`;
- `finger_indices`: `[50]`, values 0 through 4;
- `finger_names`: `[50]`;
- `anchor_types`: `[50]`, `lateral` or `bending`;
- `levels`: `[50]`, values 0 through 4;
- `trajectory_t`: `[50]`, the five level fractions;
- `target_parameters`: `[50]`;
- `observed_parameters`: `[50]`;
- `candidate_counts`: `[50]`;
- `support_counts`: `[50]`.

Ordering is thumb, index, middle, ring, pinky; within each finger, five
lateral levels followed by five bending levels. Metadata stores schema
version, raw filename and SHA-256, side, topology, all effective parameters,
tolerance fallback history, and source frame indices.

The miner refuses to overwrite outputs unless `--overwrite` is explicit.
It writes:

- `outputs/anchors/anchors_human_<side>_report.json`;
- `outputs/anchors/anchors_human_<side>_report.html`.

The HTML report shows per-finger/type candidate histograms, robust endpoints,
five target and observed values, support counts, fallback history, and 3D
views of the five selected real hand poses. Plotly is embedded once.

## Robot Five-Level Trajectories

`geort/anchor/anchor_spec.py` defines only level fractions and robot
trajectories. It contains no fixed human-angle table.

- Lateral: sample the target MCP2 joint from its URDF lower to upper limit at
  the five level fractions; other target-finger joints remain neutral.
- Non-thumb bending: find the feasible scalar interval for
  `[alpha, beta1, beta2, beta3] = [0, b, b, b/2]` by intersecting the URDF
  limits, then sample that interval at the five level fractions.
- Thumb bending: densely sample a feasible neutral-to-flexion joint path,
  evaluate exact URDF FK, compute TIP arc length, and select five qpos values
  at uniform arc fractions.
- All non-target joints stay at neutral, defined as zero clipped to the URDF
  limits.

Human angles are never copied into robot joints. Pairing is exclusively by
`finger_index`, `anchor_type`, and `level`.

## P2/P3 Generation

`geort/anchor/interpolate.py` remains a generic piecewise-linear
`[5, D] -> [K, D]` utility because it handles both XYZ and qpos paths.

`geort/anchor/generate_robot_anchors.py`:

1. validates the 50-row mined-human contract;
2. constructs the matching five robot qpos values for each finger/type;
3. interpolates human TIP and robot qpos paths to `K=50` lateral or
   `K=100` bending;
4. evaluates exact URDF FK for the target robot TIP;
5. writes `data/anchors_<hand>.npz`;
6. atomically updates the selected prepared manifest;
7. writes separate human and robot trajectory panels for visual review.

Final arrays are:

- `human_points`: `[750, 3]`;
- `robot_points`: `[750, 3]`;
- `finger_indices`: `[750]`;
- `anchor_types`: `[750]`;
- `trajectory_t`: `[750]`;
- `source_sparse_indices`: `[750, 2]`.

Counts are 250 lateral plus 500 bending rows.

## Trainer Integration

The prepared-data loader accepts sparse anchor rows plus `finger_indices`
and normalizes each row with the selected finger's human/robot statistics.

For each batch it scatters the human point into a zero `[B,5,3]` TIP tensor,
runs the existing independent per-finger IK plus frozen FK, gathers only the
target finger, and computes `L_align` against the paired robot point.

The Step 5 P-Chamfer, distance, and motion losses and their weights remain
unchanged. `L_align` has weight 1 when anchors are present.

## Removal of the Superseded Workflow

The uncommitted live collector and its tests are deleted:

- `geort/anchor/collect_human_anchors.py`;
- `tests/test_collect_human_anchors.py`.

The current fixed-angle human task builder and tests are replaced by the
robot-only trajectory specification. No SAPIEN viewer, stdin interaction,
HTS socket, partial progress file, H3 collection, or H4 collection remains
in Step 6/7.

## Verification and Acceptance

Automated tests cover:

- 21-point topology and palm-frame rigid-transform invariance;
- synthetic known `alpha/beta` angles;
- asymmetric robust endpoint extraction;
- geometric levels differing from empirical-CDF quantiles;
- distinct real-frame medoids, deterministic tie-breaking, and bounded
  candidate sets;
- tolerance relaxation records and hard failure after exhaustion;
- thumb arc-length levels;
- exactly 50 ordered sparse human records;
- robot URDF feasibility and level order;
- generic 50/100 interpolation and final 750-row ordering;
- atomic report/output/manifest behavior;
- sparse finger-indexed trainer normalization and gather behavior.

Step 6 mining acceptance requires:

- at least five candidates for every finger/type after recorded fallback;
- monotonic observed level parameters;
- five distinct source frames per finger/type;
- a complete JSON and HTML report for human inspection.

Post-generation acceptance additionally requires correct 750-row counts and
visually coherent human/robot level ordering. Post-training acceptance remains
`L_align` decreasing without degrading the other three losses, followed by
replay showing more stable ambiguous poses than the Step 5 checkpoint.

Human work is report inspection only. There is no data-collection step.
