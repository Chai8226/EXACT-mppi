# 3D smoothness telemetry in scenario summaries

Status: implemented

## Parent

.scratch/3d-web-replay-and-scenario-suite/PRD.md

## What to build

Add 3D smoothness telemetry to scenario runs so control quality can be judged from command and state histories rather than animation playback. The telemetry should work for deterministic scenario summaries and be suitable for later display in Offline Web replay.

The first metric set should quantify command changes and trajectory smoothness in a stable, documented way. It should be computed from the executed command and state histories produced by the Python scenario loop.

## Acceptance criteria

- [x] Scenario summaries include finite command smoothness metrics derived from `[vx, vy, vz, wz]` history.
- [x] Scenario summaries include finite trajectory smoothness metrics derived from executed `[x, y, z, yaw]` state history.
- [x] Smoothness telemetry is independent of viewer interpolation or frame rate.
- [x] Focused tests cover metric behavior using simple known histories.
- [x] The open-track baseline summary includes the new telemetry fields.

## Blocked by

- .scratch/3d-web-replay-and-scenario-suite/issues/01-config-driven-3d-scenario-runner-open-track.md
