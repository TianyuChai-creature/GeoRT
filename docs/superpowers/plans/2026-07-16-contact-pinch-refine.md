# Contact Pinch Refine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add opt-in runtime contact-probability blending with deterministic, bounded analytic-FK pinch refinement.

**Architecture:** `geort/contact/runtime.py` owns classifier checkpoint loading, 6D hand-base features, probability selection, and projected joint refinement. `geort/export.py` owns opt-in integration after the existing mapper returns qpos; `off` returns the original qpos object/value path. The optimiser receives only the selected thumb and target-finger eight joint indices and uses analytic FK plus projection to physical joint bounds.

**Tech Stack:** NumPy, PyTorch CPU, existing `AnalyticFK`, existing `HandFormatter`, pytest.

---

### Task 1: Runtime loading and trigger tests

**Files:**
- Create: `geort/contact/runtime.py`
- Create: `tests/test_contact_runtime.py`

- [x] Write failing tests for: a checkpoint's four pair scalers/models load; raw `[21,3]` hand-base landmarks produce `[4,6]`; `w=clip((p-p_lo)/(p_hi-p_lo),0,1)`; highest probability wins ties by pair order; off returns q_map bitwise.
- [x] Run the new test file and observe missing-module failure.
- [x] Implement `ContactRefiner.load`, `probabilities`, `select_trigger`, and `blend_qpos` without optimiser logic.
- [x] Run the new test file.
- [x] Commit runtime classifier/trigger layer.

### Task 2: Bounded analytic-FK refinement tests

**Files:**
- Modify: `geort/contact/runtime.py`
- Modify: `tests/test_contact_runtime.py`

- [x] Write failing tests for selected-pair-only eight-DOF updates, hard joint-limit projection, zero target distance reduction, deterministic repeated output, and non-selected joints equal q_map.
- [x] Run tests and observe missing refinement method.
- [x] Implement fixed-count projected Adam in physical qpos with objective `||tip_thumb-tip_finger||^2 + 0.1||q-q_map||^2`; use `AnalyticFK` and physical limits. No other loss/trainer changes.
- [x] Run tests.
- [x] Commit bounded refiner.

### Task 3: Export/realtime integration and evidence

**Files:**
- Modify: `geort/export.py`
- Modify: `geort/mocap/hts_realtime_inference.py`
- Modify: `tests/test_contact_runtime.py`
- Create: `docs/contact_right_d1_training_report.md`

- [x] Write failing tests for `--contact_refine off` exact qpos preservation and explicit on-path parameters.
- [x] Run tests and observe absent CLI/integration.
- [x] Add `--contact_refine`, checkpoint, probability, target-distance, regularisation and fixed-step arguments; default off. Route raw 21 landmarks to refiner only on on-path.
- [x] Run deterministic CPU acceptance: bounds, distances, q deltas, timing, off regression, continuous D1 segment duty cycle.
- [x] Commit integration and evidence.
