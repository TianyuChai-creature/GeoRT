# L_motion local-frame execution plan

Goal: ship two commits without changing the default global CUDA behavior.

## Commit A: explicit device

- [ ] Add failing tests for parser default cuda, CPU selection, and CPU FK device placement.
- [ ] Replace hidden training-path CUDA placement and CUDA-only RNG/cache assumptions with an explicit torch.device.
- [ ] Run target tests, then run the explicit 50-step CUDA baseline/candidate pair and compare logs byte-for-byte.
- [ ] Commit only device plumbing and its tests.

## Commit B: local T(x)

- [ ] Add failing pure tests for human frames, degeneracy fallback, robot constants, optional FK rotations, and SAPIEN rotation parity.
- [ ] Add geort.motion_frames with the fixed geometry, thresholds, DIP sign convention, literals, cache schema, and shared validation helpers.
- [ ] Add point-cloud and anchor rotation fields only for newly generated artifacts; require the fields only in local readers.
- [ ] Add P-Chamfer optional argmin output and frame-aware motion loss, retaining the existing global branch verbatim.
- [ ] Add trainer motion_frame wiring, run-local human-frame cache, metadata/startup fields, and isolated anchor behavior tests.
- [ ] Make LMC import the shared frame helper; add the explicit CPU diagnostic CLI.
- [ ] Run CPU tests, CPU diagnostics, and the explicit CUDA global 50-step comparison against Commit A.
- [ ] Commit sources, tests, specification, and this plan as Commit B.
