# Realtime C2b Provenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make C2b seed 42 the strictly verified realtime default while persisting full session provenance and operator-requested frozen-frame evidence.

**Architecture:** The ledger remains the single source of checkpoint provenance; its generator receives a named extra checkpoint record for C2b. The realtime entrypoint keeps the existing safety path unchanged, records startup metadata in its session summary, and delegates frozen-frame persistence to `SessionRecorder`.

**Tech Stack:** Python 3.12, NumPy, pytest, existing GeoRT/SAPIEN realtime runtime.

---

### Task 1: Register the C2b runtime checkpoint in the provenance ledger

**Files:**
- Modify: `geort/mocap/generate_realtime_checkpoint_ledger.py`
- Test: `tests/test_realtime_provenance.py`

- [ ] **Step 1: Write the failing test**

```python
def test_build_ledger_adds_explicit_c2b_record(tmp_path):
    ledger = build_ledger(final_matrix, tmp_path, extra_checkpoints={
        "c2b_s42": tmp_path / "checkpoint" / "c2b_s42",
    })
    assert ledger["runs"]["c2b_s42"]["last_pth_sha256"] == expected_sha
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/creature/Desktop/GeoRT/.venv/bin/python -m pytest tests/test_realtime_provenance.py::test_build_ledger_adds_explicit_c2b_record -q`

Expected: FAIL because `build_ledger` has no `extra_checkpoints` parameter.

- [ ] **Step 3: Write the minimal implementation**

```python
def build_ledger(final_matrix, repo_root, *, extra_checkpoints=()):
    # Preserve final-matrix entries and append each named checkpoint by its own metadata.
```

The generated C2b record must include `checkpoint`, `last_pth_sha256`, `motion_frame`, and `anchor` exactly as `verify_archived_checkpoint` consumes them.

- [ ] **Step 4: Run the test to verify it passes**

Run the command from Step 2. Expected: PASS.

### Task 2: Persist frozen frames and required session metadata

**Files:**
- Modify: `geort/mocap/realtime_runtime.py`
- Test: `tests/test_realtime_safety.py`

- [ ] **Step 1: Write failing tests**

```python
recorder.freeze_frame(
    raw_points=np.zeros((21, 3)), normalized_tips=np.zeros((5, 3)),
    mapped_qpos=np.zeros(20), output_qpos=np.zeros(20), timestamp_s=1.0,
)
path = recorder.close(counters=RealtimeCounters(), extra_summary={"smoothing_alpha": None})
assert (path / "frozen_frames.npz").exists()
assert json.loads((path / "summary.json").read_text())["smoothing_alpha"] is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/creature/Desktop/GeoRT/.venv/bin/python -m pytest tests/test_realtime_safety.py -q`

Expected: FAIL because `freeze_frame` does not exist.

- [ ] **Step 3: Write the minimal implementation**

```python
def freeze_frame(self, *, timestamp_s, raw_points, normalized_tips, mapped_qpos, output_qpos):
    self._frozen_frames.append({...})
```

Write the same arrays to `frozen_frames.npz` in `close`; retain existing `frames.npz`, counters, safety behavior, and summary merge semantics.

- [ ] **Step 4: Run the tests to verify they pass**

Run the command from Step 2. Expected: PASS.

### Task 3: Switch realtime default and connect the `F` freeze action

**Files:**
- Modify: `geort/mocap/hts_realtime_inference.py`
- Modify: `geort/mocap/verify_realtime_c2.py`
- Test: `tests/test_hts_realtime_inference.py`

- [ ] **Step 1: Write failing tests**

```python
args = realtime.build_arg_parser().parse_args([])
assert args.checkpoint.endswith("custom_right_2026-07-17_12-21-39_c2b_s42")
assert args.freeze_key == "f"
```

Exercise a fake viewer key event and assert that `SessionRecorder.freeze_frame` receives the current raw input, normalized input, mapped qpos, and safe output qpos.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/creature/Desktop/GeoRT/.venv/bin/python -m pytest tests/test_hts_realtime_inference.py -q`

Expected: FAIL because the default remains C2 and no freeze key exists.

- [ ] **Step 3: Write the minimal implementation**

Set the C2b directory as the default in both realtime CLI tools. Add the exact expected SHA constant and reject a mismatching startup provenance. Add `--freeze-key` defaulting to `f`; edge-trigger it in the viewer loop after a mapped/safe frame exists. Add `smoothing_alpha`, checkpoint SHA, `git rev-parse HEAD`, and `" ".join(sys.argv)` to `SessionRecorder.close(extra_summary=...)`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/creature/Desktop/GeoRT/.venv/bin/python -m pytest tests/test_hts_realtime_inference.py tests/test_realtime_safety.py tests/test_realtime_provenance.py -q`

Expected: PASS.

### Task 4: Generate provenance and run the offline parity gate

**Files:**
- Generate: `outputs/final_matrix/checkpoint_hashes.json`
- Generate: `outputs/realtime_sessions/<timestamp>/`

- [ ] **Step 1: Generate the C2b-inclusive ledger**

Run: `/home/creature/Desktop/GeoRT/.venv/bin/python -m geort.mocap.generate_realtime_checkpoint_ledger --extra-checkpoint c2b_s42=checkpoint/custom_right_2026-07-17_12-21-39_c2b_s42`

- [ ] **Step 2: Run 1000-frame parity**

Run: `/home/creature/Desktop/GeoRT/.venv/bin/python -m geort.mocap.verify_realtime_c2 --checkpoint checkpoint/custom_right_2026-07-17_12-21-39_c2b_s42 --frames 1000 --seed 42`

Report the printed maximum physical-qpos difference in radians.

- [ ] **Step 3: Commit source and tests**

Run: `git add geort/mocap tests docs/superpowers/plans && git commit -m "feat: default realtime runtime to audited c2b"`

