# Offline Web replay export schema and writer

Status: implemented

## Parent

.scratch/3d-web-replay-and-scenario-suite/PRD.md

## What to build

Add an Offline Web replay export path for deterministic yaw-only 3D scenario runs. Python should complete the simulation and write world-frame replay data that a passive browser viewer can load without reconstructing controller-local frame internals.

The replay data should separate static `scene` data from dynamic `frames` data. It should include enough information to inspect obstacle points, reference path, local plan windows, executed path, optimal trajectory, T-shaped robot volume, commands, clearance, goal distance, and smoothness telemetry.

## Acceptance criteria

- [x] A scenario run can export a replay artifact containing static scene data and per-frame data.
- [x] Replay scene data includes scenario identity, coordinate conventions, obstacle points, reference path, and robot volume config.
- [x] Replay frames include robot state, executed path, local plan in global coordinates, optimal trajectory in global coordinates when available, command, clearance, goal distance, and smoothness telemetry.
- [x] Replay export uses world coordinates for viewer-facing geometry and does not require the viewer to reconstruct local-frame controller inputs.
- [x] Tests verify finite replay data shape and required fields for at least the open-track and T-shaped narrow-gap scenarios.

## Blocked by

- .scratch/3d-web-replay-and-scenario-suite/issues/01-config-driven-3d-scenario-runner-open-track.md
- .scratch/3d-web-replay-and-scenario-suite/issues/02-3d-smoothness-telemetry-summaries.md
- .scratch/3d-web-replay-and-scenario-suite/issues/03-authoritative-t-shaped-3d-robot-volume-scenario.md
