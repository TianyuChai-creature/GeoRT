# custom_right Anchor QA Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a static, reproducible Markdown quality-assurance report and five PNG figures for the existing `custom_right`/`hts_right.npy` anchor data.

**Architecture:** Add one CPU-only report module that reads the D1 recording, human sparse anchors, canonical 750-row parity qpos, current-run normalization contract, and persisted parity result. It derives every metric from these inputs, writes a JSON metric record plus Markdown/PNG presentation, and fails explicitly when a required contract is absent or inconsistent.

**Tech Stack:** Python 3.12 (main worktree virtual environment), NumPy, Matplotlib, existing `geort.anchor` geometry/compat/normalization modules.

---

### Task 1: Define report inputs and metric helpers with unit tests

**Files:**

- Create: `tests/test_anchor_qa_report.py`
- Create: `geort/anchor/qa_report.py`

- [ ] **Step 1: Write failing tests for report input contracts and trajectory metrics**

```python
def test_validate_parity_composition_rejects_noncanonical_counts():
    bundle = {"finger_indices": np.zeros(749, np.int64), "anchor_types": np.array(["lateral"] * 749)}
    with pytest.raises(ValueError, match="750"):
        validate_parity_composition(bundle)

def test_trajectory_quality_detects_uniform_forward_sequence():
    points = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    result = trajectory_quality(points)
    assert result["all_direction_dots_positive"] is True
    assert result["step_ratio_max_min"] == pytest.approx(1.0)

def test_span_ratio_uses_each_side_normalization_stats():
    human = np.array([[0., 0., 0.], [2., 0., 0.]])
    robot = np.array([[0., 0., 0.], [4., 0., 0.]])
    assert normalized_span_ratio(human, robot, {"center": [0, 0, 0], "scale": 2}, {"center": [0, 0, 0], "scale": 4}) == pytest.approx(1.0)
```

- [ ] **Step 2: Run the tests and confirm they fail because `qa_report` does not exist**

Run: `PYTHONPATH=. /home/creature/Desktop/GeoRT/.venv/bin/python -m pytest tests/test_anchor_qa_report.py -q`

Expected: collection failure for `geort.anchor.qa_report`.

- [ ] **Step 3: Implement pure metric helpers**

```python
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
ANCHOR_TYPES = ("lateral", "bending")

def trajectory_quality(points: np.ndarray) -> dict[str, float | bool]:
    steps = np.diff(np.asarray(points, dtype=np.float64), axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    dots = np.einsum("ij,ij->i", steps[:-1], steps[1:])
    return {
        "step_ratio_max_min": float(lengths.max() / lengths.min()) if lengths.min() > 1e-12 else float("inf"),
        "all_direction_dots_positive": bool(np.all(dots > 0.0)),
        "min_direction_dot": float(dots.min()) if dots.size else float("inf"),
    }
```

Implement `validate_parity_composition` to require exactly 50 lateral rows and 100 bending rows for every finger, and normalize human and robot points through `normalize_finger_points` with their respective statistics before measuring L1→L5 Euclidean span.

- [ ] **Step 4: Run the focused tests**

Run: `PYTHONPATH=. /home/creature/Desktop/GeoRT/.venv/bin/python -m pytest tests/test_anchor_qa_report.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the helper layer**

```bash
git add geort/anchor/qa_report.py tests/test_anchor_qa_report.py
git commit -m "feat: add anchor QA metric helpers"
```

### Task 2: Assemble all A–D evidence into a machine-readable report record

**Files:**

- Modify: `geort/anchor/qa_report.py`
- Modify: `tests/test_anchor_qa_report.py`

- [ ] **Step 1: Write a failing end-to-end fixture test**

```python
def test_build_report_record_contains_all_requested_sections(tmp_path):
    record = build_report_record(input_paths_for_fixture(tmp_path))
    assert set(record) >= {"decision", "human_self_check", "robot_and_pairing", "contract"}
    assert len(record["decision"]["parameter_percentiles"]) == 10
    assert len(record["decision"]["span_ratios"]) == 10
    assert record["robot_and_pairing"]["parity"]["overall"]["max_m"] < 1e-3
```

- [ ] **Step 2: Run the fixture test and confirm it fails**

Run: `PYTHONPATH=. /home/creature/Desktop/GeoRT/.venv/bin/python -m pytest tests/test_anchor_qa_report.py::test_build_report_record_contains_all_requested_sections -q`

Expected: failure because `build_report_record` is absent.

- [ ] **Step 3: Implement record construction using the existing artifacts**

`build_report_record` must:

1. load raw D1 and recompute the valid-frame action parameter distributions with the mining geometry; report per-level percentile for non-thumb lateral alpha, non-thumb bending beta1, and thumb main-trajectory arc fraction;
2. load the 50 sparse human levels and the canonical parity qpos, select `trajectory_t = [0, .25, .5, .75, 1]` for each robot group, evaluate their TIPs via `make_analytic_tip_callback`, normalize each side with the matching `normalization.json` subsection, and calculate L1→L5 spans and ratios;
3. read human candidate/support counts and compute bending coupling residuals `abs(MCP1-PIP)` and `abs(DIP-PIP/2)` on selected frames; mark candidate counts `<10`, nonmonotonic paths, zero intervals, duplicate points, or an interval ratio above `3` as warnings;
4. report robot joint/TIP interval tables, thumb arc equal-spacing relative deviation, parity composition, interpolated human path step uniformity and positive consecutive-dot checks;
5. copy the persisted FK gate from `custom_right_fk_parity.json`, require `overall.max_m < 1e-3`, and add `git rev-parse HEAD`, coordinate declarations, normalization path, `human_data_source`, and every input/output path to the contract section.

The code must never call the SAPIEN backend for report metrics or rerun the parity gate. It must use `make_analytic_tip_callback` and the offsets in the same custom_right config field as parity.

- [ ] **Step 4: Run the complete QA module tests**

Run: `PYTHONPATH=. /home/creature/Desktop/GeoRT/.venv/bin/python -m pytest tests/test_anchor_qa_report.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the evidence record**

```bash
git add geort/anchor/qa_report.py tests/test_anchor_qa_report.py
git commit -m "feat: collect custom-right anchor QA evidence"
```

### Task 3: Render Markdown tables and static five-finger 3D figures

**Files:**

- Modify: `geort/anchor/qa_report.py`
- Modify: `tests/test_anchor_qa_report.py`
- Create at runtime: `outputs/anchors/qa_custom_right/anchor_qa_report.md`
- Create at runtime: `outputs/anchors/qa_custom_right/metrics.json`
- Create at runtime: `outputs/anchors/qa_custom_right/figures/{thumb,index,middle,ring,pinky}.png`

- [ ] **Step 1: Write failing output tests**

```python
def test_write_report_creates_md_json_and_all_finger_figures(tmp_path):
    paths = write_report(record, tmp_path)
    assert paths.markdown.read_text(encoding="utf-8").startswith("# custom_right Anchor QA")
    assert json.loads(paths.metrics.read_text())["contract"]["coordinate_space"]["units"] == "m"
    assert {p.stem for p in paths.figures} == {"thumb", "index", "middle", "ring", "pinky"}
```

- [ ] **Step 2: Run the output test and confirm it fails**

Run: `PYTHONPATH=. /home/creature/Desktop/GeoRT/.venv/bin/python -m pytest tests/test_anchor_qa_report.py::test_write_report_creates_md_json_and_all_finger_figures -q`

Expected: failure because `write_report` is absent.

- [ ] **Step 3: Implement rendering and CLI**

Render A–D as Markdown tables, using `<mark>…</mark>` for warnings. Each per-finger Matplotlib 3D PNG has two panels: D1 target-finger TIP cloud plus its human level-1…5 points, and analytic-FK robot reachable sample cloud (the canonical group trajectory) plus robot level-1…5 points. Add CLI defaults fixed to `custom_right`, `data/hts_right.npy`, `data/anchors_human_right.npz`, `outputs/anchors/parity_qpos.npz`, `outputs/anchors/custom_right_fk_parity.json`, and the seed42 normalization path; a differing `--normalization-path` remains explicit and recorded.

- [ ] **Step 4: Run rendering tests**

Run: `MPLBACKEND=Agg PYTHONPATH=. /home/creature/Desktop/GeoRT/.venv/bin/python -m pytest tests/test_anchor_qa_report.py -q`

Expected: all tests pass and no GUI/GPU process is created.

- [ ] **Step 5: Commit renderer and CLI**

```bash
git add geort/anchor/qa_report.py tests/test_anchor_qa_report.py
git commit -m "feat: render static custom-right anchor QA report"
```

### Task 4: Produce and verify the requested report

**Files:**

- Create at runtime: `outputs/anchors/qa_custom_right/*`

- [ ] **Step 1: Generate without overwriting source anchors**

```bash
MPLBACKEND=Agg PYTHONPATH=. /home/creature/Desktop/GeoRT/.venv/bin/python -m geort.anchor.qa_report \
  --hand custom_right \
  --human-data data/hts_right.npy \
  --human-anchors data/anchors_human_right.npz \
  --parity-qpos outputs/anchors/parity_qpos.npz \
  --parity-report outputs/anchors/custom_right_fk_parity.json \
  --normalization-path checkpoint/custom_right_2026-07-16_10-08-30_seed42_null_v3_full/normalization.json \
  --output-dir outputs/anchors/qa_custom_right
```

- [ ] **Step 2: Verify every hard report contract**

```bash
test -s outputs/anchors/qa_custom_right/anchor_qa_report.md
test -s outputs/anchors/qa_custom_right/metrics.json
test "$(find outputs/anchors/qa_custom_right/figures -name '*.png' | wc -l)" -eq 5
rg -n '1.2082900982071858e-07|250 lateral \+ 500 bending|hand-base|human_data_source' outputs/anchors/qa_custom_right/anchor_qa_report.md
```

Expected: all commands succeed; report explicitly carries all required A–D evidence and the referenced prior FK gate.

- [ ] **Step 3: Commit only source/tests, not generated data unless requested**

```bash
git add geort/anchor/qa_report.py tests/test_anchor_qa_report.py
git commit -m "test: verify custom-right anchor QA report"
```

