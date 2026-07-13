# AnyDexRT Step 6 D1-Mined Anchors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mine 50 real sparse human anchors from an existing D1 HTS recording,
pair them with robot-range levels, generate 750 training pairs, and enable
sparse finger-indexed `L_align`.

**Architecture:** Pure palm-frame geometry and mining modules provide
deterministic, testable behavior. A thin CLI loads D1, writes atomic NPZ/JSON
and a Plotly report. Robot trajectories are defined independently from human
angles, then exact URDF FK and the existing generic interpolation produce the
trainer-ready bundle.

**Tech Stack:** Python 3.12, NumPy, PyTorch, SAPIEN FK, Plotly, pytest.

---

## Construction Status at Checkpoint 73780db+

Recorded on 2026-07-13 after the user requested an early checkpoint:

- Task 1 is complete and passed specification and code-quality review.
  The superseded live collector is absent; generic `[5,D]` interpolation
  remains and has 14 passing focused tests.
- Task 2 is complete and passed specification and code-quality review.
  Palm alignment and angle extraction process all 234,114 right-hand D1
  frames in about 1.83 seconds; 37 focused tests pass.
- Task 3 has an implemented, tested core for robust endpoints, bounded local
  medoids, monotonic distinct selection, and candidate fallback. Its original
  focused suite reached 54 passing tests, but specification review found two
  unresolved blockers:
  1. `robust_angle_targets` still permits caller-defined level fractions;
     it must expose only fixed `[0, 0.25, 0.5, 0.75, 1]` levels.
  2. Exact-medoid distance squaring can overflow for finite descriptors near
     `1e200`; uniformly scale descriptors before pairwise subtraction.
- Task 3 must add RED tests for those two blockers, implement the fixes, and
  repeat specification and quality review before it is accepted.
- Tasks 4 through 8 have not started. No human-anchor NPZ, HTML report,
  robot-paired anchors, manifest update, trainer integration, or new training
  artifact has been produced.

### Task 1: Remove Superseded Collection and Preserve Generic Interpolation

**Files:**
- Delete: `geort/anchor/collect_human_anchors.py`
- Delete: `tests/test_collect_human_anchors.py`
- Modify: `geort/anchor/__init__.py`
- Keep: `geort/anchor/interpolate.py`
- Keep: `tests/test_anchor_interpolate.py`

- [x] **Step 1: Remove the live collection files**

Delete the uncommitted collector and its tests. Remove exports of
`SparseAnchorTask` and `build_sparse_anchor_tasks`; later tasks add the
new robot trajectory API.

- [x] **Step 2: Verify generic interpolation**

Run:

```bash
PYTHONPATH=/usr/lib/python3/dist-packages:. \
  /home/creature/Desktop/GeoRT/.venv/bin/python -m pytest -q \
  tests/test_anchor_interpolate.py
```

Expected: 14 tests pass, including `[5,20] -> [100,20]` support.

- [x] **Step 3: Check removal scope**

Run `rg -n "collect_human_anchors|SparseAnchorTask" geort tests docs`.
Only the historical design diff or explicit removal documentation may match.

### Task 2: Palm Frame and Human Finger Angle Geometry

**Files:**
- Create: `geort/anchor/human_geometry.py`
- Create: `tests/test_anchor_human_geometry.py`

- [x] **Step 1: Write failing topology and palm-frame tests**

Tests require:

```python
aligned, valid = align_hts_to_palm(frames)
assert aligned.shape == (batch, 21, 3)
assert valid.shape == (batch,)
```

Use synthetic frames to prove translation, rotation, and uniform-scale
invariance. Reject non-`[T,21,3]`, non-finite, zero palm width, and collinear
palm axes.

- [x] **Step 2: Confirm RED**

Run only `tests/test_anchor_human_geometry.py`; expect missing module.

- [x] **Step 3: Implement palm alignment**

Define the fixed 21-landmark topology. Build wrist origin, wrist-to-middle
longitudinal axis, orthogonalized pinky-to-index lateral axis, cross-product
normal, and index-to-pinky scale. Return aligned float64 points plus a validity
mask without mutating input.

- [x] **Step 4: Write failing known-angle tests**

Generate articulated synthetic index-through-pinky chains with known
`alpha/beta1/beta2/beta3`. Require:

```python
angles = estimate_finger_angles(aligned)
assert angles["alpha"].shape == (batch, 5)
assert angles["beta"].shape == (batch, 5, 3)
```

Check sign, zero pose, known bends, rigid-transform invariance, and thumb
lateral output.

- [x] **Step 5: Implement angle estimation**

Use normalized segment vectors, clipped dot products, and `atan2`. Return
angles and per-finger validity masks. Thumb beta values may be diagnostic but
must not drive thumb bending levels.

- [x] **Step 6: Verify Task 2**

Run the geometry tests and `python -m compileall -q geort/anchor`.

### Task 3: Geometric Levels, Deterministic Medoids, and Fallbacks

**Files:**
- Create: `geort/anchor/mining.py`
- Create: `tests/test_anchor_mining.py`

- [x] **Step 1: Write failing robust-range tests**

Require `robust_angle_targets(values, endpoint_quantiles=(0.02,0.98))` to
return five values uniformly spaced between robust endpoints. Use a
neutral-heavy distribution to prove results differ from empirical CDF
quantiles and preserve asymmetric limits.

- [x] **Step 2: Confirm RED and implement robust targets**

Reject degenerate/non-finite ranges and invalid quantiles. Return endpoints,
targets, and endpoint support metadata.

- [x] **Step 3: Write failing medoid tests**

Require a deterministic API:

```python
selection = select_level_medoids(
    parameters,
    descriptors,
    source_indices,
    targets,
    min_support=5,
    max_candidates=256,
)
```

Test exact medoid choice, source-index tie-breaking, distinct source frames,
band expansion history, monotonic selected parameters, and hard failure when
five distinct frames cannot be selected.

- [x] **Step 4: Implement bounded local medoids**

Use target-distance ordering to cap each neighborhood, chunked pairwise
Euclidean sums for exact medoids, deterministic tie-breaking, and the
`[1,2,4,8]` band schedule.

- [x] **Step 5: Write failing candidate-filter tests**

Test lateral straightness, non-thumb bending
`max(|beta1-beta2|, |beta1-2*beta3|)`, alpha-near-zero filtering, tolerance
factors `[1,1.5,2,3,4]`, first-success selection, and recorded exhaustion.

- [x] **Step 6: Implement candidate filtering**

Keep finite/geometry validity and endpoint clipping outside the fallback.
Return candidate masks, counts, effective tolerances, and full fallback
history.

- [x] **Step 7: Verify Task 3 implementation**

Run `tests/test_anchor_mining.py` plus the geometry tests.

- [ ] **Step 8: Resolve Task 3 review blockers**

Lock `robust_angle_targets` to the approved five fractions and stabilize
Euclidean medoid distances for extreme finite descriptors. Re-run the focused
suite and both review gates.

### Task 4: Thumb Main-Trajectory Mining and 50-Row Human Contract

**Files:**
- Modify: `geort/anchor/mining.py`
- Modify: `tests/test_anchor_mining.py`

- [ ] **Step 1: Write failing thumb trajectory tests**

Create a curved synthetic thumb TIP path with dense neutral dwell and
outliers. Require five distinct real source indices at uniformly spaced arc
fractions, deterministic output, at least five populated bins, and report
metadata including PCA explained variance.

- [ ] **Step 2: Implement thumb trajectory**

PCA-order the robust thumb cloud, compute deterministic medoids in 64 ordered
bins, construct the medoid polyline, and choose real frames around five
uniform cumulative-arc targets.

- [ ] **Step 3: Write failing full mining-contract test**

Use synthetic 21-point frames containing all ten motions. Require exactly 50
rows ordered by finger, type, and level with fields:

```text
human_frames, human_points, source_indices, finger_indices, finger_names,
anchor_types, levels, trajectory_t, target_parameters, observed_parameters,
candidate_counts, support_counts
```

- [ ] **Step 4: Implement `mine_human_anchor_records`**

Compose geometry, candidate filters, geometric targets, medoids, and thumb
arc mining. Validate monotonicity and distinct frames per group before
returning arrays plus JSON-serializable diagnostics.

- [ ] **Step 5: Verify Tasks 2-4**

Run all anchor geometry/mining/interpolation tests together.

### Task 5: Mining CLI, Atomic Outputs, and Inspection Report

**Files:**
- Create: `geort/anchor/mine_human_anchors.py`
- Create: `tests/test_mine_human_anchors.py`

- [ ] **Step 1: Write failing CLI-core tests**

Test NPY shape validation, SHA-256 metadata, default paths
`data/anchors_human_<side>.npz`, explicit overwrite protection, atomic NPZ
and JSON replacement, and report payload counts.

- [ ] **Step 2: Implement output core and CLI**

CLI arguments include:

```text
--input data/hts_right.npy
--hand-side right
--output data/anchors_human_right.npz
--report-dir outputs/anchors
--endpoint-low 0.02
--endpoint-high 0.98
--straight-tol-deg 15
--alpha-zero-tol-deg 10
--coupling-tol-deg 20
--overwrite
```

Runtime files are ignored by Git. Do not update the prepared manifest yet.

- [ ] **Step 3: Write failing report tests**

Require one histogram and five-pose 3D view per finger/type, target/observed
markers, counts, endpoints, fallback history, source indices, and one embedded
Plotly bundle.

- [ ] **Step 4: Implement JSON and HTML reports**

Keep human pose views inspectable and report file size bounded. The CLI exits
nonzero without writing final outputs if any acceptance check fails.

- [ ] **Step 5: Verify CLI without mining production D1**

Run CLI `--help` and a synthetic end-to-end fixture.

### Task 6: Robot Range Levels and Exact-FK Pair Generation

**Files:**
- Create: `geort/anchor/anchor_spec.py`
- Create: `tests/test_anchor_spec.py`
- Create: `geort/anchor/generate_robot_anchors.py`
- Create: `tests/test_generate_robot_anchors.py`
- Modify: `geort/anchor/__init__.py`

- [ ] **Step 1: Write failing robot trajectory tests**

Require five lateral qpos values across MCP2 limits, five non-thumb bending
values across the feasible intersection for `[0,b,b,b/2]`, neutral
non-target joints, material limit validation, and no fixed human-angle table.

- [ ] **Step 2: Implement robot trajectory specification**

Expose level fractions and pure lateral/non-thumb qpos constructors. Keep
robot values independent from mined human angle values.

- [ ] **Step 3: Write failing thumb FK arc tests**

With a fake FK callback, require dense feasible thumb qpos sampling followed
by five uniform TIP-arc selections with exact endpoints.

- [ ] **Step 4: Implement thumb trajectory and paired builder**

`build_paired_anchors` validates 50 mined rows, interpolates human XYZ and
robot qpos by group, calls target-finger FK, and produces exactly 750 rows.

- [ ] **Step 5: Write failing manifest/report tests**

Require atomic manifest update preserving unrelated fields, schema/source
metadata, counts, and separate human/robot Plotly panels.

- [ ] **Step 6: Implement generator CLI**

Accept `--hand`, `--human-anchors`, `--manifest`, `--output`,
`--report`, and `--overwrite`. Initialize configured keypoints once and
use exact URDF FK.

- [ ] **Step 7: Verify Task 6**

Run anchor spec, generator, mining, and interpolation tests.

### Task 7: Sparse Finger-Indexed Trainer Integration

**Files:**
- Modify: `geort/trainer.py`
- Modify: `tests/test_anydexrt_trainer.py`

- [ ] **Step 1: Write failing sparse loader tests**

Change `PreparedTrainingData.anchor_points` to carry human `[N,3]`, robot
`[N,3]`, and integer `finger_indices [N]`. Test per-row normalization with
distinct finger statistics and reject mismatched/non-finite/out-of-range
arrays.

- [ ] **Step 2: Write failing sparse gather tests**

Require:

```python
mapped = map_sparse_anchors(human, finger_indices, tip_count=5, map_tips=fake)
assert mapped.shape == (batch, 3)
```

Use independent fake finger outputs to prove only the indexed finger affects
the result.

- [ ] **Step 3: Implement loader, scatter, and gather**

Scatter each row into `[B,5,3]`, call the existing mapper, and gather the
target finger. Include indices in the anchor `TensorDataset`.

- [ ] **Step 4: Integrate `L_align`**

Compute alignment on gathered `[B,3]` points with batch 32 and weight 1.
Do not alter P-Chamfer, distance, motion, their weights, or inference
normalization.

- [ ] **Step 5: Run trainer tests and one-epoch synthetic smoke**

Require finite Align history, `anchors_enabled=true`, and unchanged
checkpoint contracts.

### Task 8: Real D1 Mining, Documentation, and Final Verification

**Files:**
- Modify: `docs/AnyDexRT 改造执行手册.md`
- Modify: `data/README.md`
- Modify: `README.md`

- [ ] **Step 1: Update Step 6/7 workflow text**

Remove H3/H4 collection instructions. Document D2 mining now and leave Step 7
auto-label implementation unchecked pending its separate plan.

- [ ] **Step 2: Run the right-hand D1 miner**

```bash
GEORT_PYTHON=/home/creature/Desktop/GeoRT/.venv/bin/python
PYTHONPATH=. "$GEORT_PYTHON" -m geort.anchor.mine_human_anchors \
  --input data/hts_right.npy \
  --hand-side right
```

Require ten groups, each with at least five candidates, five distinct
monotonic levels, and complete JSON/HTML reports.

- [ ] **Step 3: Run P3 generation**

```bash
PYTHONPATH=. "$GEORT_PYTHON" -m geort.anchor.generate_robot_anchors \
  --hand custom_right \
  --human-anchors data/anchors_human_right.npz \
  --manifest data/hts_right_prepared.json
```

Require 750 rows and a manifest anchor entry.

- [ ] **Step 4: Run fresh full verification**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 UV_CACHE_DIR=/tmp/uv-cache \
  uv run --no-project \
  --python /home/creature/Desktop/GeoRT/.venv/bin/python \
  --with pytest python -m pytest -q
/home/creature/Desktop/GeoRT/.venv/bin/python -m compileall -q geort tests
git diff --check
```

- [ ] **Step 5: Report artifacts and request human report inspection**

Provide paths and candidate/fallback/count summary. Do not start training
until the mined report is visually approved.

- [ ] **Step 6: Request commit/push approval**

Commit and push Step 6 code only after explicit user approval.
