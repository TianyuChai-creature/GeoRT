# Hand Manifest M1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add manifest-driven intake and hand-agnostic FK parity with a declarative FK gate, preserving legacy parity invocation compatibility.

**Architecture:** `geort/pipeline/manifest.py` owns YAML decoding, required fields, path resolution and `get_joint_limits` validation. `geort/pipeline/gates.py` owns `fk_parity_max_m` comparison and a serializable gate record. `geort.anchor.parity` gains `--manifest` and writes qpos/report under the manifest output root while leaving the explicit legacy arguments available. Hand YAML and gate YAML are declarative inputs, not Python defaults.

**Tech Stack:** Python 3.12, PyYAML, NumPy, existing SAPIEN/analytic FK, pytest.

---

### Task 1: Manifest and gate tests

**Files:**
- Create: `tests/test_pipeline_manifest.py`
- Create: `tests/test_pipeline_gates.py`

- [ ] **Step 1: Write the failing manifest test**

```python
def test_load_hand_manifest_resolves_paths_and_limits(tmp_path: Path) -> None:
    manifest_path = tmp_path / "hand.yaml"
    manifest_path.write_text(
        "hand_id: demo\n"
        "urdf: assets/custom_left/URDF_L.urdf\n"
        "hts: data/hts_left.npy\n"
        "hand_config: custom_left\n"
        "output_root: outputs/demo\n"
    )
    manifest = load_hand_manifest(manifest_path)
    assert manifest.hand_id == "demo"
    assert manifest.output_root == Path("outputs/demo")
    lower, upper = manifest.joint_limits()
    assert lower.shape == upper.shape == (20,)
```

- [ ] **Step 2: Run manifest test to verify RED**

Run: `/home/creature/Desktop/GeoRT/.venv/bin/python -m pytest tests/test_pipeline_manifest.py -q`  
Expected: import error for `geort.pipeline.manifest`.

- [ ] **Step 3: Write the failing gate test**

```python
def test_fk_parity_gate_is_inclusive_at_limit(tmp_path: Path) -> None:
    spec = load_gate_spec(tmp_path / "gates.yaml")
    assert evaluate_fk_parity_gate({"overall": {"max_m": 1e-6}}, spec)["passed"] is True
    assert evaluate_fk_parity_gate({"overall": {"max_m": 1.1e-6}}, spec)["passed"] is False
```

- [ ] **Step 4: Run gate test to verify RED**

Run: `/home/creature/Desktop/GeoRT/.venv/bin/python -m pytest tests/test_pipeline_gates.py -q`  
Expected: import error for `geort.pipeline.gates`.

### Task 2: Minimal manifest and gate implementation

**Files:**
- Create: `geort/pipeline/__init__.py`
- Create: `geort/pipeline/manifest.py`
- Create: `geort/pipeline/gates.py`
- Create: `configs/hands/custom_right.yaml`
- Create: `configs/hands/custom_left.yaml`
- Create: `configs/gates.yaml`

- [ ] **Step 1: Implement manifest loader**

```python
@dataclass(frozen=True)
class HandManifest:
    hand_id: str
    urdf: Path
    hts: Path
    hand_config: str
    output_root: Path
    anchor_bundle: Path | None = None

    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        config = get_config(self.hand_config)
        return tuple(np.asarray(value, dtype=np.float64) for value in get_joint_limits(config))
```

The loader must require the five mandatory YAML keys, require `output_root == outputs/<hand_id>`, verify that the config's `urdf_path` resolves to the declared `urdf`, and never inspect `config["joint"]`.

- [ ] **Step 2: Implement gate loader/evaluator**

```python
def evaluate_fk_parity_gate(report: Mapping[str, Any], spec: GateSpec) -> dict[str, Any]:
    observed = float(report["overall"]["max_m"])
    return {"name": "fk_parity_max", "observed_m": observed,
            "limit_m": spec.fk_parity_max_m, "passed": observed <= spec.fk_parity_max_m}
```

- [ ] **Step 3: Run focused tests to verify GREEN**

Run: `/home/creature/Desktop/GeoRT/.venv/bin/python -m pytest tests/test_pipeline_manifest.py tests/test_pipeline_gates.py -q`  
Expected: both test modules pass.

### Task 3: Manifest-driven parity entry point and tests

**Files:**
- Modify: `geort/anchor/parity.py`
- Modify: `tests/test_pipeline_manifest.py`

- [ ] **Step 1: Write the failing entry-point test**

```python
def test_manifest_parity_writes_output_and_docs_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # monkeypatch parity write/compare helpers; use a tmp manifest and gate file.
    report = parity.main(["--manifest", str(manifest_path), "--gates", str(gates_path)])
    assert (tmp_path / "outputs/demo/parity/parity_report.json").exists()
    assert (tmp_path / "docs/reports/demo_fk_parity.md").exists()
    assert report["gate"]["passed"] is True
```

- [ ] **Step 2: Run the entry-point test to verify RED**

Run: `/home/creature/Desktop/GeoRT/.venv/bin/python -m pytest tests/test_pipeline_manifest.py::test_manifest_parity_writes_output_and_docs_archive -q`  
Expected: argparse rejects `--manifest`.

- [ ] **Step 3: Add `--manifest` flow**

`--manifest` must load the hand manifest, use `output_root/parity/parity_qpos.npz` and `output_root/parity/parity_report.json`, apply `configs/gates.yaml`, write the numeric report plus `gate` record, and archive a Markdown report. `--hand`, `--human-anchors`, `--parity-qpos`, and `--report` remain accepted only when no manifest is supplied; remove their right-hand defaults and make legacy required inputs explicit.

- [ ] **Step 4: Run focused tests to verify GREEN**

Run: `/home/creature/Desktop/GeoRT/.venv/bin/python -m pytest tests/test_pipeline_manifest.py tests/test_pipeline_gates.py -q`  
Expected: all pass.

### Task 4: Real parity evidence and read-only attachment answers

**Files:**
- Create: `docs/reports/custom_right_fk_parity_m1.{json,md}`
- Create: `docs/reports/custom_left_fk_parity_m1.{json,md}`

- [ ] **Step 1: Run right manifest entry**

Run: `/home/creature/Desktop/GeoRT/.venv/bin/python -m geort.anchor.parity --manifest configs/hands/custom_right.yaml --gates configs/gates.yaml`  
Expected: report has qpos rows `750`; max is at or below `1e-6 m`.

- [ ] **Step 2: Run left manifest entry**

Run: `/home/creature/Desktop/GeoRT/.venv/bin/python -m geort.anchor.parity --manifest configs/hands/custom_left.yaml --gates configs/gates.yaml`  
Expected: report has qpos rows `750`; if gate fails, stop before any later work and report the raw analytic/SAPIEN results and URDF suspicion chain.

- [ ] **Step 3: Record source answers**

Read and quote: `build_target_cloud.py` rest/motion use; `hts_right_mocap.py` left/right reader, landmark layout and mirroring behavior; `mine_human_anchors.py` hand-side effects.

### Task 5: Final verification and commit

**Files:**
- All M1 files above only.

- [ ] **Step 1: Run focused tests**

Run: `/home/creature/Desktop/GeoRT/.venv/bin/python -m pytest tests/test_pipeline_manifest.py tests/test_pipeline_gates.py -q`

- [ ] **Step 2: Run full suite**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/creature/Desktop/GeoRT/.venv/bin/python -m pytest -q`

- [ ] **Step 3: Verify scope and commit**

Run: `git diff --check && git status --short`; inspect that no `data/`, `checkpoint/`, anchor, trainer or evaluator code changed. Commit with `feat: add manifest-driven FK parity station`.
