# Quest C2 Realtime Design

## Scope

Drive the finalized C2 seed-42 checkpoint from Quest right-hand landmarks. The
implementation changes only realtime/export-side code and tests. Trainer, loss,
checkpoints, and data artifacts remain read-only.

## Architecture

`geort.mocap.hts_realtime_inference` remains the public CLI. A focused realtime
runtime component owns checkpoint provenance checks, offline D1 parity, safety
state, and per-session recording. It obtains mapping through the existing
export/evaluation loading path and calls `normalize_finger_points`; it does not
reimplement model loading or normalization.

The default checkpoint resolves to
`custom_right_2026-07-16_22-04-19_c2_s42`. `--checkpoint` may select another
checkpoint. At startup, the runtime reads checkpoint metadata, computes the
weight SHA256, and rejects a C2-default provenance mismatch against the final
matrix input records.

## Safety and operation

Each valid frame follows: raw Quest points -> existing right-hand/hand-base
preprocessing -> existing normalized mapping -> optional ContactRefiner ->
per-joint clamp and rate limit -> startup/recovery ramp. Invalid input retains
the preceding output and increments a NaN counter. A >200 ms input timeout
freezes output and causes the next valid frames to ramp again. Keyboard e-stop
freezes output.

Stage 1 uses the SAPIEN mirror with this same runtime. Stage 2 uses Quest input
with contact off. Stage 3 only enables contact refinement; it changes no mapping
or safety path.

## Evidence and tests

A D1 1000-frame offline runner calls the realtime mapping path and the
assessment mapping path, reporting max absolute qpos error and enforcing
`<= 1e-6 rad`. Unit tests cover provenance rejection, clamp/rate/ramp behavior,
NaN hold, watchdog freeze, and session schema. Every session stores raw points,
normalized input, mapped/refined/safe outputs, timestamps, stage timing, and
counter/latency summaries under `outputs/realtime_sessions/<timestamp>/`.
