# Scenario-suite baseline report for current 3D controller

Status: implemented

## Parent

.scratch/3d-web-replay-and-scenario-suite/PRD.md

## What to build

Generate the first 3D replay baseline for the current yaw-only 3D controller across the static scenario suite. The report should aggregate scenario summaries and identify which scenarios pass, fail, collide, miss the goal, or show poor smoothness before any MPPI parameter or cost-function tuning begins.

This slice should produce evidence for follow-up tuning work rather than changing controller behavior.

## Acceptance criteria

- [x] The full static scenario suite can be run as a baseline command without opening a browser.
- [x] The baseline report includes reached-goal, collision, final distance, minimum clearance, step count, command smoothness, and trajectory smoothness for each scenario.
- [x] The report includes an aggregate pass/fail overview across the static scenario suite.
- [x] The report identifies scenarios that need follow-up tuning without changing MPPI parameters or critic behavior in this slice.
- [x] Replay artifacts can be emitted for baseline runs when requested.

## Blocked by

- .scratch/3d-web-replay-and-scenario-suite/issues/02-3d-smoothness-telemetry-summaries.md
- .scratch/3d-web-replay-and-scenario-suite/issues/04-static-3d-scenario-suite-expansion.md
- .scratch/3d-web-replay-and-scenario-suite/issues/05-offline-web-replay-export-schema-writer.md
