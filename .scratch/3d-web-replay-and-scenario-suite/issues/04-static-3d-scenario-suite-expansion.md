# Static 3D scenario suite expansion

Status: implemented

## Parent

.scratch/3d-web-replay-and-scenario-suite/PRD.md

## What to build

Expand the yaw-only 3D scenario suite beyond the open-track and narrow-gap baselines with deterministic static scenarios that isolate specific navigation capabilities. The first suite should include vertical gates, T-shaped traps, and cluttered corridors, while keeping dynamic obstacles out of scope.

The suite should support selecting individual scenarios and running all static scenarios headlessly with consistent result summaries.

## Acceptance criteria

- [x] `vertical_gate_3d` can run headlessly and reports the standard scenario summary fields.
- [x] `t_shape_trap_3d` can run headlessly and reports the standard scenario summary fields.
- [x] `cluttered_corridor_3d` can run headlessly and reports the standard scenario summary fields.
- [x] The suite can run selected scenarios individually and all static scenarios as a batch.
- [x] Dynamic obstacle behavior is not introduced in this slice.

## Blocked by

- .scratch/3d-web-replay-and-scenario-suite/issues/01-config-driven-3d-scenario-runner-open-track.md
- .scratch/3d-web-replay-and-scenario-suite/issues/03-authoritative-t-shaped-3d-robot-volume-scenario.md
