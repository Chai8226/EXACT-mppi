# 迁移 3D control preference critics

Status: ready-for-agent

## Parent

.scratch/yaw-only-3d-mppi/PRD.md

## What to build

Add 3D counterparts for the control preference critics that shape command behavior without defining the main geometric objective. The 3D planner should preserve the existing configuration style where practical while extending behavior to `[vx, vy, vz, wz]`.

This slice covers prefer-forward behavior, velocity deadband behavior, and twirling behavior. Twirling should remain focused on unnecessary yaw-rate use, not path yaw alignment.

## Acceptance criteria

- [ ] Prefer-forward scoring works in the yaw-only 3D control space without introducing path-angle yaw alignment.
- [ ] Velocity deadband scoring supports 4D controls.
- [ ] Twirling scoring penalizes unnecessary yaw-rate behavior.
- [ ] Configuration names and defaults stay close to the existing 2D critic conventions where practical.
- [ ] Tests cover external scoring behavior for each 3D control preference critic.

## Blocked by

None - the 3D Core API slice is implemented and pending human review.
