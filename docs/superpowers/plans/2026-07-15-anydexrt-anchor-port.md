# AnyDexRT Anchor Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the D1-mined sparse-anchor pipeline to `AnyDexRT` for `custom_right`, generate analytic-FK anchors, and add a weight-controlled `L_align` without changing the existing mining, interpolation, or regularisation objectives.

**Architecture:** Copy the existing deterministic mining modules unchanged in behaviour, then adapt only their boundaries.  The paired-anchor bundle retains raw HTS TIP contexts and metric robot TIP targets with an explicit hand-base/metre contract.  At training time, the current run writes `normalization.json`, reloads that exact contract for anchors, normalizes the full human context and finger-indexed robot target, maps the context with the existing analytic FK path, and gathers the paired finger before adding `w_anchor * L_align`.

**Tech Stack:** Python 3.12, NumPy, PyTorch, pytorch-kinematics, SAPIEN (diagnostic baseline only), pytest.

---

## File map

- `geort/anchor/{__init__,interpolate,human_geometry,mining,mine_human_anchors}.py`: deterministic D1 mining and report generation copied from `anydexrt`; only the report schema gains percentile diagnostics.
- `geort/anchor/anchor_spec.py`: copied robot range trajectories; its qpos knots remain physical radians.
- `geort/anchor/generate_robot_anchors.py`: copied pairing/interpolation, adapted to evaluate qpos with `AnalyticFK` plus config TIP offsets; an explicit SAPIEN evaluator is retained only for the pre/post comparison.
- `geort/anchor/compare_robot_anchor_fk.py`: reads equal-shape analytic and SAPIEN bundles, reports per-finger and global metre errors, and fails at `1e-3` m.
- `geort/anchor/training.py`: narrow, CPU-testable bundle loader/coordinate-contract validator/normalizer for raw anchors.
- `geort/trainer.py`: creates a 32-row anchor loader after it persists normalization, evaluates the existing `fk_backend` model, and adds the weighted gathered alignment term without changing current losses or nullspace computation.
- `tests/test_anchor_*.py`, `tests/test_anchor_training.py`, `tests/test_trainer_anchors.py`: copied coverage plus target-branch contract and regression tests.
- `docs/superpowers/specs/2026-07-13-anydexrt-step6-sparse-anchors-design.md`: copied provenance design document; the migration plan is this file and does not redefine mining.

### Task 1: Port the pure mining dependencies unchanged

**Files:**
- Create: `geort/anchor/__init__.py`, `geort/anchor/interpolate.py`, `geort/anchor/human_geometry.py`, `geort/anchor/mining.py`, `geort/anchor/mine_human_anchors.py`
- Create: `tests/test_anchor_interpolate.py`, `tests/test_anchor_human_geometry.py`, `tests/test_anchor_mining.py`, `tests/test_mine_human_anchors.py`
- Copy: `docs/superpowers/specs/2026-07-13-anydexrt-step6-sparse-anchors-design.md`

- [ ] **Step 1: Copy the anchor source and original tests without edits, except imports required by the target package.**

  ```bash
  git checkout anydexrt -- geort/anchor/__init__.py geort/anchor/interpolate.py \
    geort/anchor/human_geometry.py geort/anchor/mining.py geort/anchor/mine_human_anchors.py \
    tests/test_anchor_interpolate.py tests/test_anchor_human_geometry.py \
    tests/test_anchor_mining.py tests/test_mine_human_anchors.py \
    docs/superpowers/specs/2026-07-13-anydexrt-step6-sparse-anchors-design.md
  ```

- [ ] **Step 2: Run the copied tests before adding target-specific behaviour.**

  ```bash
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /home/creature/Desktop/GeoRT/.venv/bin/python -m pytest -q \
    tests/test_anchor_interpolate.py tests/test_anchor_human_geometry.py tests/test_anchor_mining.py tests/test_mine_human_anchors.py
  ```

  Expected: the ported pure-NumPy mining contract passes; if the environment lacks pytest, record the environment failure and use the repository test interpreter selected by the user rather than changing the production code.

### Task 2: Add percentile diagnostics without changing candidate selection

**Files:**
- Modify: `geort/anchor/mining.py`
- Modify: `geort/anchor/mine_human_anchors.py`
- Test: `tests/test_mine_human_anchors.py`

- [ ] **Step 1: Write a failing report test for five selected levels against all D1 candidate parameters.**

  ```python
  payload = build_report_payload(anchors, input_path=source, hand_side="right", config={})
  group = payload["groups"][0]
  assert group["selected_percentiles"] == [0.0, 25.0, 50.0, 75.0, 100.0]
  assert group["distribution_parameter"] in {"alpha", "beta1", "thumb_tip_arc"}
  ```

- [ ] **Step 2: Run the single test and confirm it fails because the report field is absent.**

  ```bash
  PYTHONPATH=. /home/creature/Desktop/GeoRT/.venv/bin/python -m pytest -q tests/test_mine_human_anchors.py -k percentiles
  ```

- [ ] **Step 3: Persist full candidate-parameter arrays in mining diagnostics and render a table in JSON/HTML.**

  The added diagnostic must compute each selected value's inclusive empirical percentile in that finger/type's original finite D1 candidate parameter vector.  It must not alter filters, fallback factors, endpoints, selected source indices, five levels, or the 50/100 interpolation counts.  Each group records the parameter name, candidate count, selected values, and five percentages in level order.

- [ ] **Step 4: Re-run mining/report tests.**

  ```bash
  PYTHONPATH=. /home/creature/Desktop/GeoRT/.venv/bin/python -m pytest -q tests/test_anchor_mining.py tests/test_mine_human_anchors.py
  ```

### Task 3: Port robot trajectories and adapt qpos FK to AnalyticFK

**Files:**
- Create: `geort/anchor/anchor_spec.py`, `geort/anchor/generate_robot_anchors.py`, `geort/anchor/compare_robot_anchor_fk.py`
- Create: `tests/test_anchor_spec.py`, `tests/test_generate_robot_anchors.py`, `tests/test_compare_robot_anchor_fk.py`

- [ ] **Step 1: Copy the pure robot trajectory specification and its existing tests.**

  ```bash
  git checkout anydexrt -- geort/anchor/anchor_spec.py tests/test_anchor_spec.py
  ```

- [ ] **Step 2: Write failing tests for qpos-to-analytic TIP evaluation and the 1 mm comparison gate.**

  ```python
  analytic = evaluate_analytic_tip_fk(qpos, config)
  assert analytic.shape == (len(qpos), 5, 3)
  assert np.isfinite(analytic).all()
  report = compare_robot_anchor_bundles(legacy_path, analytic_path)
  assert report["max_error_m"] < 1e-3
  ```

- [ ] **Step 3: Implement only the boundary adapter.**

  Build `AnalyticFK(config["urdf_path"], lower, upper, tip_offsets=tip_info["offset"])`, normalize physical generator qpos with the same lower/upper limits, call it under `torch.no_grad()`, and return metres in the robot hand base frame.  Validate the five config finger names and the five offset rows before generation.  Do not use the neural FK model.

  The generator CLI defaults to `--fk-backend analytic`; `--fk-backend sapien` is permitted only to create the legacy comparison bundle using `HandKinematicModel.keypoint_from_qpos` with identical link offsets.  Both bundles include `coordinate_frame="hand_base"`, `units="m"`, source SHA-256, qpos, and `fk_backend` metadata.

- [ ] **Step 4: Implement the comparison gate and test it.**

  The comparator requires matching `robot_qpos`, `finger_indices`, and `trajectory_t`, calculates `||tip_analytic-tip_sapien||_2` for every one of 750 rows, writes JSON with global and per-finger max/mean/p95 errors, and raises `RuntimeError` if the maximum is at least `0.001` m.

  ```bash
  PYTHONPATH=. /home/creature/Desktop/GeoRT/.venv/bin/python -m pytest -q \
    tests/test_anchor_spec.py tests/test_generate_robot_anchors.py tests/test_compare_robot_anchor_fk.py
  ```

### Task 4: Normalize raw anchors from the persisted current-run contract

**Files:**
- Create: `geort/anchor/training.py`, `tests/test_anchor_training.py`

- [ ] **Step 1: Write failing tests for coordinate assertions and indexed normalization.**

  ```python
  loaded = load_raw_anchor_training_points(anchor_path, normalization_path, finger_names)
  assert loaded.human_contexts.shape == (2, 5, 3)
  assert loaded.robot_targets.shape == (2, 3)
  assert loaded.finger_indices.tolist() == [0, 1]
  with pytest.raises(ValueError, match="hand_base"):
      load_raw_anchor_training_points(wrong_frame_path, normalization_path, finger_names)
  ```

- [ ] **Step 2: Run the test and confirm the loader is missing.**

  ```bash
  PYTHONPATH=. /home/creature/Desktop/GeoRT/.venv/bin/python -m pytest -q tests/test_anchor_training.py
  ```

- [ ] **Step 3: Implement a raw-anchor loader that reads the current `normalization.json`.**

  Require bundle fields `human_tip_contexts [N,5,3]`, `robot_points [N,3]`, `finger_indices [N]`, and JSON metadata declaring human and robot `coordinate_frame="hand_base"` and `units="m"`.  Require the normalization contract to describe the identical ordered five tip finger names.  Normalize all five human context columns with `normalization["human"]`; normalize each robot row using `normalization["robot"][finger_names[finger_index]]`.  Return float32 normalized arrays and int64 indices.  This is the sole anchor normalization path.

- [ ] **Step 4: Re-run loader tests.**

  ```bash
  PYTHONPATH=. /home/creature/Desktop/GeoRT/.venv/bin/python -m pytest -q tests/test_anchor_training.py
  ```

### Task 5: Add `L_align` to the current trainer without perturbing other losses

**Files:**
- Modify: `geort/trainer.py`
- Create: `tests/test_trainer_anchors.py`

- [ ] **Step 1: Write failing CPU-level tests for indexed gather, CLI defaults, metadata, and disabled anchors.**

  ```python
  assert build_arg_parser().parse_args([]).w_anchor == 1.0
  selected = gather_anchor_fingers(torch.tensor([[[1.,0,0],[2.,0,0]]]), torch.tensor([1]))
  assert torch.equal(selected, torch.tensor([[2.,0,0]]))
  assert metadata["anchors"]["weight"] == 1.0
  ```

- [ ] **Step 2: Run the test and confirm it fails for the absent anchor API.**

  ```bash
  PYTHONPATH=. /home/creature/Desktop/GeoRT/.venv/bin/python -m pytest -q tests/test_trainer_anchors.py
  ```

- [ ] **Step 3: Implement the isolated anchor path.**

  Add `--anchor_path` (default `None`) and `--w_anchor` (default `1.0`).  After computing and saving the run's existing `normalization.json`, load the optional raw anchor bundle from that exact file.  Build `TensorDataset(human_contexts, robot_targets, finger_indices)` and a shuffled `DataLoader(..., batch_size=32)`.  On every ordinary training batch, consume one anchor batch cyclically, call the already selected `fk_model(ik_model(human_contexts))`, normalize neither side again, gather the requested TIP, and add `w_anchor * F.mse_loss(gathered, robot_targets)` to `loss`.

  Preserve the existing Chamfer, distance, curvature, motion, synergy, pinch, collision, MCP1 prior, and nullspace expressions verbatim.  Add an `Align` metric only to logging.  Add metadata `{enabled, path, sha256, count, batch_size: 32, weight, fk_backend, normalization_path}` under `anchors` and include `w_anchor` in the saved weight dictionary.

- [ ] **Step 4: Re-run the trainer-anchor tests and existing nullspace tests.**

  ```bash
  PYTHONPATH=. /home/creature/Desktop/GeoRT/.venv/bin/python -m pytest -q \
    tests/test_trainer_anchors.py tests/test_analytic_fk.py tests/test_deliverables.py
  ```

### Task 6: Generate and validate only custom_right anchors

**Files:**
- Output (ignored): `data/anchors_human_right.npz`, `data/anchors_custom_right_sapien.npz`, `data/anchors_custom_right.npz`, `outputs/anchors/*`

- [ ] **Step 1: Mine right-hand human anchors and inspect the new percentile table.**

  ```bash
  PYTHONPATH=. /home/creature/Desktop/GeoRT/.venv/bin/python -m geort.anchor.mine_human_anchors \
    --input data/hts_right.npy --hand-side right --output data/anchors_human_right.npz \
    --report-dir outputs/anchors --overwrite
  ```

  Require 50 rows, ten groups, five source frames per group, monotonic level parameters, and all ten report percentile rows.

- [ ] **Step 2: Generate the SAPIEN reference and analytic production bundle from the same human anchors.**

  ```bash
  PYTHONPATH=. /home/creature/Desktop/GeoRT/.venv/bin/python -m geort.anchor.generate_robot_anchors \
    --hand custom_right --human-anchors data/anchors_human_right.npz \
    --output data/anchors_custom_right_sapien.npz --fk-backend sapien --overwrite
  PYTHONPATH=. /home/creature/Desktop/GeoRT/.venv/bin/python -m geort.anchor.generate_robot_anchors \
    --hand custom_right --human-anchors data/anchors_human_right.npz \
    --output data/anchors_custom_right.npz --fk-backend analytic --overwrite
  ```

- [ ] **Step 3: Gate production anchors on the FK comparison.**

  ```bash
  PYTHONPATH=. /home/creature/Desktop/GeoRT/.venv/bin/python -m geort.anchor.compare_robot_anchor_fk \
    --legacy data/anchors_custom_right_sapien.npz --analytic data/anchors_custom_right.npz \
    --output outputs/anchors/custom_right_fk_comparison.json --max-error-mm 1.0
  ```

  Stop before any training if the command reports a maximum error of 1.0 mm or greater; compare qpos order, hand-base transform, URDF joint names, and local TIP offsets before attempting another generation.

- [ ] **Step 4: Run all CPU-only anchor tests; do not start a training process or use GPU.**

  ```bash
  PYTHONPATH=. /home/creature/Desktop/GeoRT/.venv/bin/python -m pytest -q \
    tests/test_anchor_interpolate.py tests/test_anchor_human_geometry.py tests/test_anchor_mining.py \
    tests/test_mine_human_anchors.py tests/test_anchor_spec.py tests/test_generate_robot_anchors.py \
    tests/test_compare_robot_anchor_fk.py tests/test_anchor_training.py tests/test_trainer_anchors.py
  ```
