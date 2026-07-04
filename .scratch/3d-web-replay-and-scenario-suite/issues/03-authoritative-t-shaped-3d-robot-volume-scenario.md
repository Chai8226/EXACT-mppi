# Authoritative T-shaped 3D robot volume scenario

Status: implemented

## Parent

.scratch/3d-web-replay-and-scenario-suite/PRD.md

## What to build

Add a T-shaped 3D robot volume as the authoritative body for yaw-only 3D collision evaluation and scenario export. Demonstrate it through a narrow-gap 3D scenario that exercises the non-convex robot volume rather than a decorative display-only model.

The same volume configuration should be used by collision checks and included in scenario results for later Web rendering.

## Acceptance criteria

- [x] A T-shaped 3D robot volume can be configured as a box-union volume for scenario runs.
- [x] The configured T-shaped volume is used by collision evaluation during the run.
- [x] The configured T-shaped volume is included in exported/run result data for visualization consumers.
- [x] A deterministic narrow-gap scenario exercises the T-shaped volume and reports reached-goal, collision, clearance, and step metrics.
- [x] Tests verify that the scenario collision geometry and exported robot volume come from the same authoritative configuration.

## Blocked by

- .scratch/3d-web-replay-and-scenario-suite/issues/01-config-driven-3d-scenario-runner-open-track.md
