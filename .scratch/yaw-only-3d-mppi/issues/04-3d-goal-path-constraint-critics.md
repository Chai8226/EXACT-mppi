# 迁移 3D goal、path、constraint 相关 critics

Status: ready-for-agent

## Parent

.scratch/yaw-only-3d-mppi/PRD.md

## What to build

Add 3D counterparts for the core goal, path, and constraint critics in the yaw-only 3D planner. Position-based scoring should use `x`, `y`, and `z`; yaw-specific scoring should compare only yaw. Goal yaw should only apply when the 3D position is near the goal. Path alignment should be position-only and must not align intermediate yaw to the reference path.

Do not add `PathAngleCritic3D`.

## Acceptance criteria

- [ ] Constraint scoring works with 4D yaw-only 3D controls.
- [ ] Goal position scoring uses 3D distance to `[x, y, z]`.
- [ ] Goal yaw scoring compares only yaw and is inactive outside its configured near-goal threshold.
- [ ] Path alignment scoring uses position-only 3D distance and does not constrain intermediate yaw.
- [ ] Path following measures progress along a 3D reference path.
- [ ] There is no `PathAngleCritic3D` in the 3D critic set.
- [ ] Tests cover the external behavior of 3D goal, goal yaw, path alignment, path following, and constraint scoring.

## Blocked by

None - the 3D Core API slice is implemented and pending human review.
