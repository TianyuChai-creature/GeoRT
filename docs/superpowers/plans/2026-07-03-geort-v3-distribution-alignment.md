# GeoRT V3 Distribution Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the v3 distribution-alignment proposal from `docs/3946d46a-c46d-42ea-a76f-608e54bf9134_GeoRT_分布对齐改造_v3：从数据采集到验收的技术方案.pdf`.

**Architecture:** Add a Quest3/HTS session collector, build a human-shaped robot target cloud from rest/motion captures, switch IK chamfer targets between uniform and human-shaped clouds, then run A/B metrics. The first version keeps loss code, IK/FK model structure, URDF, and joint limits unchanged.

**Tech Stack:** Python 3.12, NumPy, PyTorch, SAPIEN 3, hand-tracking-sdk, pytest-style tests.

---

### Phase 0: Quest3/HTS Session Collection

**Files:**
- Create: `geort/mocap/collect_hts_session.py`
- Test: `tests/test_collect_hts_session.py`

- [ ] Add a timed session collector that records rest and motion in one process.
- [ ] Reuse `geort.mocap.hts_right_mocap.iter_hts_points` and its `[T, 21, 3]` frame validation conventions.
- [ ] Require the operator to press Enter before rest and again before motion.
- [ ] Print the PDF motion checklist before motion capture starts.
- [ ] Save `data/hts_{side}_{session_id}_rest.npy`, `data/hts_{side}_{session_id}.npy`, and `data/hts_{side}_{session_id}.json`.
- [ ] Record segment start/end times, requested/actual durations, frame counts, estimated FPS, device/operator metadata, transport settings, and finite-value validation.
- [ ] Verify with tests that do not require Quest3 hardware.
- [ ] Reminder before archival: ask user approval before `git commit`.

### Phase 1: Human-Shaped Target Cloud

**Files:**
- Create: `geort/mocap/build_target_cloud.py`
- Optional split: `geort/mocap/target_cloud_angles.py`, `geort/mocap/target_cloud_mold.py`, `geort/mocap/target_cloud_debug.py`
- Test: `tests/test_build_target_cloud.py`

- [ ] Implement `extract_angles(frames) -> [T, n_dof]`, driven by `config["joint_order"]`.
- [ ] Implement flexion, AA, and thumb CMC angle proxies.
- [ ] Compute rest noise floor, rest anchor, motion P2/P98 endpoints, and pin validity.
- [ ] Build per-joint piecewise-linear mold `M`.
- [ ] Apply `M` to motion frames, density-cap q-space voxels without flattening the distribution, run robot FK, and save `qpos` plus `keypoint` in `RobotKinematicsDataset` format.
- [ ] Emit `mold.json`, angle arrays, histograms, mold plots, and uniform-vs-human target cloud plots.
- [ ] Verify mold behavior and dataset compatibility with tests.
- [ ] Reminder before archival: ask user approval before `git commit`.

### Phase 2: Trainer Chamfer Target Switch

**Files:**
- Modify: `geort/trainer.py`
- Create: `geort/utils/hash_utils.py`
- Test: `tests/test_trainer_chamfer_target.py`

- [ ] Add `--chamfer_target {uniform,human}`, `--chamfer_target_path`, and `--mold_path`.
- [ ] Keep `uniform` behavior unchanged.
- [ ] Make `human` load `data/{hand}_humanshaped.npz` or the explicit target path, failing loudly if missing.
- [ ] Keep neural FK training on the uniform full-domain dataset.
- [ ] Write target cloud hash, mold hash, human data path, loss weights, epoch count, and CLI settings to checkpoint metadata.
- [ ] Verify uniform compatibility, human target selection, missing-file errors, and metadata writes with tests.
- [ ] Reminder before archival: ask user approval before `git commit`.

### Phase 3: A/B Acceptance Metrics

**Files:**
- Create: `geort/mocap/metrics_evaluation.py`
- Test: `tests/test_metrics_evaluation.py`

- [ ] Compute signed gain, bilateral saturation rate, rest offset, and pinch failure rate.
- [ ] Enforce the PDF baseline gate: run A/uniform first and stop claims if A does not reproduce pathology.
- [ ] Compare B/human against PDF acceptance lines.
- [ ] Save unified metrics JSON with data hash, mold hash, target cloud hash, checkpoints, recipe, and all metrics.
- [ ] Verify metric math and gate behavior with fake models.
- [ ] Reminder before archival: ask user approval before `git commit`.

### Execution Order

- [ ] Implement and test Phase 0.
- [ ] Collect a real Quest3/HTS rest+motion session.
- [ ] Implement and test Phase 1.
- [ ] Generate and inspect the human-shaped target cloud debug artifacts.
- [ ] Implement and test Phase 2.
- [ ] Train A/uniform and B/human with the same data, FK, recipe, and seed.
- [ ] Implement and test Phase 3.
- [ ] Run metrics and archive only after user approval.
