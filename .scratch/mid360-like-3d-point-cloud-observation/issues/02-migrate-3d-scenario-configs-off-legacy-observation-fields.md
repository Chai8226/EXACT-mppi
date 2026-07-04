# Migrate 3D scenario configs off legacy observation fields

Status: ready-for-agent

## Parent

.scratch/mid360-like-3d-point-cloud-observation/PRD.md

## What to build

Migrate the built-in yaw-only 3D scenarios from Range-based local 3D observation configuration to the MID-360-like sensor configuration. Sensor range should live under the top-level sensor configuration, and the controller point budget should live under controller configuration. The new 3D scenario path should no longer depend on simulation-level oracle fields or global sampled obstacle points for controller perception.

This slice should leave the scenarios runnable under the new perception model while preserving the scenario suite's role as the main headless evaluation surface.

## Acceptance criteria

- [ ] Built-in 3D scenario configs use a top-level MID-360-like sensor section for perception range, FOV, angular-grid density, and disabled noise/dropout defaults.
- [ ] Built-in 3D scenario configs no longer use simulation-level observation range for the new perception path.
- [ ] Built-in 3D scenario configs use controller configuration for controller obstacle point budget.
- [ ] The scenario runner reads sensor range from sensor configuration and point budget from controller configuration.
- [ ] Legacy global sampled obstacle points are not required for controller perception in migrated scenarios.
- [ ] Headless scenario-suite tests cover at least one migrated built-in scenario using the new sensor configuration.
- [ ] Configuration validation or errors make missing/unsupported sensor configuration clear for the new path.

## Blocked by

- .scratch/mid360-like-3d-point-cloud-observation/issues/01-mid360-like-observed-cloud-tracer-bullet.md
