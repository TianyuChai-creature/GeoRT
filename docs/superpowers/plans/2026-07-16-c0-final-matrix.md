# C0 Baseline and Final Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a compatibility-preserving C0 configuration path, verify target-cloud rotations, then run the requested C0–C3 seed matrix and fixed evaluation protocol.

**Architecture:** `geort/trainer.py` will parse an optional YAML configuration before normal argparse processing; explicit CLI values override YAML fields. `chamfer_mode=partial` preserves the established one-way loss and `bidirectional` uses the existing symmetric Chamfer function. Matrix and evaluation artifacts remain under `outputs/final_matrix/`; finalized inputs are read-only.

**Tech Stack:** Python argparse, PyYAML, PyTorch, existing analytic/neural FK, pytest, existing target-cloud generator.

---

### Task 1: C0 compatibility CLI and metadata

**Files:**
- Modify: `geort/trainer.py`
- Create: `configs/geort_equiv.yaml`
- Create: `tests/test_trainer_c0_config.py`

- [ ] Write failing tests for `partial` default, bidirectional loss selection, YAML loading, explicit CLI precedence, and emitted resolved configuration.
- [ ] Run the new test file and observe the missing `--config`/`--chamfer_mode` failure.
- [ ] Add a YAML pre-parser; set parser defaults from YAML; retain explicit CLI precedence; record resolved values in startup JSON and metadata.
- [ ] Route partial mode through the existing `partial_chamfer_distance` call unchanged; route bidirectional mode through `chamfer_distance`.
- [ ] Add `configs/geort_equiv.yaml` with the requested C0 values, including `w_distance: 0.0` (the trainer's canonical spelling of task-text `w_dist`).
- [ ] Run the new tests and the existing trainer unit tests.

### Task 2: Part A blocking evidence

**Files:**
- Create: `outputs/final_matrix/part_a_gate.json`
- Create: `outputs/final_matrix/part_a_*.log`

- [ ] Run 50-step default CUDA training at the reference commit and current C0 commit with identical explicit arguments; compare loss records line-by-line.
- [ ] Run 50-step bidirectional C0 smoke; record finite-loss and NaN counters.
- [ ] Run YAML config dump; compare every resolved field with `configs/geort_equiv.yaml`.
- [ ] Stop matrix execution if any of the three recorded gate booleans is false.

### Task 3: Rotation target-cloud parity

**Files:**
- Create: `data/custom_right_with_link_rotation.npz`
- Create: `outputs/final_matrix/target_cloud_rotation_parity.json`

- [ ] Re-run the exact finalized target-cloud protocol to the new output path.
- [ ] Compare `qpos` and every existing keypoint byte-for-byte with `data/custom_right.npz`; record only the added `link_rotation` field.
- [ ] Make C3 conditional on this parity record without overwriting any finalized asset.

### Task 4: Matrix launch and evaluation

**Files:**
- Create: `outputs/final_matrix/launch_matrix.py`
- Create: `outputs/final_matrix/final_matrix.md`
- Create: `outputs/final_matrix/final_matrix.json`

- [ ] Launch C0–C3, seed 42 then 123, sequentially with all supplied values explicit and per-run logs/config dumps.
- [ ] Continue to the next run on a training-process failure without parameter changes or retries.
- [ ] Evaluate completed checkpoints with the fixed D1/RandomState(42) protocol and store raw seed-separated results.
- [ ] Add the perturbation and C3 contact rows; emit only numeric tables and machine-readable JSON.
