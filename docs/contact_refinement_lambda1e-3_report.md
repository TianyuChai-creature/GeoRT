# Contact refinement lambda `1e-3` rerun

Base contact-refinement commit: `aed1ef6`.

Step-7 contact labeling/training commit: `c050d38`.

The default regularisation is `1e-3` in the runtime, export API, and realtime
CLI. The default contact mode remains `off`; the CLI interface is unchanged.
All measurements use CPU, `p_lo=0.5`, `p_hi=0.8`, target distance `0 m`, and
40 projected-Adam iterations. Full numeric traces are in
`docs/contact_runtime_lambda1e-3_acceptance.json`.

| measurement | value |
| --- | ---: |
| held-out selected-pair positive frames | 6,635 |
| distance before mean / p95 (m) | 0.01585003 / 0.03369935 |
| distance after mean / p95 (m) | 0.00488012 / 0.01734427 |
| absolute q delta maximum (deg) | 20.85143 |
| absolute q delta mean (deg) | 1.11556 |
| fixed iterations | 40 |
| trace frames | 187291, 194509, 197336, 225982, 233857 |
| contact-off versus no-module output | bitwise equal |
| labeled stream segment start / length | 27036 / 1000 |
| labeled selected-pair pinch frames in stream | 1000 |
| stream trigger duty: index / middle / ring / pinky | 1.0 / 0.0 / 0.0 / 0.0 |
| stream NaN count | 0 |

The 40-step objective traces end at, respectively: `3.73537e-05`,
`6.43255e-05`, `1.68566e-05`, `5.58297e-06`, `2.12846e-05`. The full 0–40
curves are retained in the JSON archive.

## Pytest inventory

Main interpreter: `/home/creature/Desktop/GeoRT/.venv/bin/python` (CPython
3.12). The focused contact suite command was:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/creature/Desktop/GeoRT/.venv/bin/python -m pytest tests/test_contact_runtime.py tests/test_hts_realtime_inference.py -q
```

It reported `9 passed` after the lambda default change. The earlier combined
contact label/model/runtime/realtime command reported `20 passed` before this
one-line default change.

The unfiltered full suite stopped during collection with these two imports:

```text
tests/test_anchor_qa_report_integration.py: cannot import QAInputs from geort.anchor.qa_report
tests/test_anchor_qa_report_static.py: no module named geort.anchor.qa_report_static
```

With only those two files ignored, the main-interpreter suite emitted one
failure marker at approximately 32 percent but the execution host interrupted
before returning its traceback or final count. A fallback `uv run pytest` could
not be used because it selected CPython 3.13, while `open3d==0.19.0` is only
available for CPython 3.12 in this environment.
