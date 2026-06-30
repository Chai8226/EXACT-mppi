# 为 3D example 增加 matplotlib 3D 可视化与 GIF 保存

Status: ready-for-agent

## Parent

.scratch/yaw-only-3d-mppi/PRD.md

## What to build

Add lightweight matplotlib 3D visualization to the Core-only yaw-only 3D obstacle-avoidance example. The visualization should follow the current examples' style: obstacle points, global reference path, local plan, optional sampled rollouts, optimal trajectory, and current 3D robot volume should be visible with familiar color conventions.

The example should support display/render controls for headless runs and optional GIF saving for inspection.

## Acceptance criteria

- [ ] The example can draw 3D obstacle points, global reference path, local plan, optimal trajectory, and current 3D robot volume.
- [ ] Sampled rollouts can be shown through an optional flag.
- [ ] Visualization can be disabled for headless runs.
- [ ] GIF saving can be enabled without making rendering mandatory for normal tests.
- [ ] The visualization layer does not change the headless example success criteria.

## Blocked by

- .scratch/yaw-only-3d-mppi/issues/06-core-only-3d-obstacle-avoidance-example.md
