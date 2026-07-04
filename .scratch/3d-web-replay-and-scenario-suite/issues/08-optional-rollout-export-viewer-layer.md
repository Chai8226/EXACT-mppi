# Optional rollout export and viewer layer

Status: implemented

## Parent

.scratch/3d-web-replay-and-scenario-suite/PRD.md

## What to build

Add optional sampled rollout export and rendering for Offline Web replay. Rollouts should remain disabled or bounded by default so replay artifacts stay small, but they should be available when diagnosing MPPI behavior.

When enabled, rollouts should be exported in world coordinates and displayed as a toggleable viewer layer.

## Acceptance criteria

- [x] Replay export can include sampled MPPI rollouts when explicitly enabled.
- [x] Rollout export is disabled or bounded/downsampled by default to avoid very large replay artifacts.
- [x] Exported rollouts are in world coordinates and do not require local-frame reconstruction in the viewer.
- [x] The viewer renders rollouts as a distinct toggleable layer when present.
- [x] Tests verify that rollout export is optional and bounded.

## Blocked by

- .scratch/3d-web-replay-and-scenario-suite/issues/05-offline-web-replay-export-schema-writer.md
- .scratch/3d-web-replay-and-scenario-suite/issues/06-minimal-static-threejs-replay-viewer.md
