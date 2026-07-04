# Config-driven 3D scenario runner with open-track baseline

Status: implemented

## Parent

.scratch/3d-web-replay-and-scenario-suite/PRD.md

## What to build

Create the first 2D-compatible 3D workflow slice for running yaw-only 3D MPPI scenarios from configuration rather than hard-coded demo constants. The slice should include a deterministic `open_track_3d` baseline scenario that can run headlessly, drive the existing 3D controller through the Python-owned control loop, and emit a machine-readable summary.

The runner should preserve the existing control-flow shape: scenario data and reference path in world coordinates, local plan and local 3D observation points transformed into the robot-local yaw-only frame, `computeVelocityCommands`, state integration, clearance checks, and final result reporting.

## Acceptance criteria

- [x] A deterministic `open_track_3d` scenario can be selected and run headlessly from configuration.
- [x] The scenario uses the existing yaw-only 3D MPPI controller without moving simulation or control ownership outside Python.
- [x] The runner emits a machine-readable summary with reached-goal status, collision status, final distance, minimum clearance, and step count.
- [x] The runner keeps scenario/planner constants in configuration where practical instead of hard-coding the open-track run in the control loop.
- [x] Existing yaw-only 3D tests and example behavior do not regress.

## Blocked by

None - can start immediately
