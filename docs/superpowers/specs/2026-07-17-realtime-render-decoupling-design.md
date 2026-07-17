# Realtime Render Decoupling Design

## Scope

Keep the existing C2b mapper, UDP transport, contact-off behavior, and safety
controller unchanged while preventing per-frame SAPIEN rendering from pacing the
receive/map/safety loop.  The temporary safety cap remains 10 rad/s.

## Design

The mapping loop will consume the latest received frame, run normalization,
mapping and safety, then publish the latest output qpos to a small render
mailbox.  It will not call `viewer_env.update()` per accepted input frame.

Rendering remains on the owner thread and is scheduled at a configurable fixed
cadence (initially 30 Hz).  The renderer consumes only the newest qpos, records
the render timestamp for that qpos, polls the existing freeze/e-stop keys, and
does not feed a blocking operation back into mapping cadence.  This avoids
moving the existing SAPIEN/OpenGL context across threads.

## Safety and recording

The existing `RealtimeSafetyController` remains the only component that clamps,
ramps, rate-limits, watchdog-freezes, or e-stops commands.  The mapper publishes
only qpos already accepted by that controller.  A session continues to contain
`t_recv`, `t_start`, `t_map`, `t_out`, and `t_render`; records are flushed after
their corresponding rendered output is observed.

## Validation

Unit coverage will establish that rendering is scheduled independently of every
accepted input frame, uses the newest published target, and preserves existing
freeze/e-stop behavior.  Existing safety and realtime tests must remain green.
The 60-second Stage-1 session will report loop intervals, segment timing,
mapped-to-output tracking, chase frames, and clean receive-dt velocity demand.

## Non-goals

No mapper, checkpoint, UDP, contact, cap-value, or renderer-architecture
redesign is included.  The implementation does not move the SAPIEN viewer or
OpenGL context to another thread.
