# Viewer inspection controls, layers, cameras, and metrics panel

Status: implemented

## Parent

.scratch/3d-web-replay-and-scenario-suite/PRD.md

## What to build

Extend the static Three.js replay viewer from a minimal playback surface into a practical inspection tool. Add layer toggles, camera presets, stable color conventions, and a compact metrics panel that exposes the replay fields needed to analyze yaw-only 3D MPPI behavior.

The viewer should remain a passive replay tool, not a scenario editor or parameter tuning UI.

## Acceptance criteria

- [x] The viewer provides toggles for obstacle points, global reference path, local plan, executed path, optimal trajectory, robot volume, and optional rollouts when present.
- [x] The viewer provides top, side, front, follow, and free camera modes.
- [x] The viewer displays current frame index, command, clearance, goal distance, and smoothness telemetry.
- [x] Reference path, local plan, executed path, optimal trajectory, and rollouts use stable distinguishable colors across replays.
- [x] The T-shaped robot yaw and body layout are visually clear during playback.
- [x] The viewer still works as a static replay tool without realtime streaming or editing controls.

## Blocked by

- .scratch/3d-web-replay-and-scenario-suite/issues/06-minimal-static-threejs-replay-viewer.md
