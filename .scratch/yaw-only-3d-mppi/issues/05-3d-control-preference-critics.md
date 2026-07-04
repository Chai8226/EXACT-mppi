# 迁移 3D control preference critics

Status: implemented

## Parent

.scratch/yaw-only-3d-mppi/PRD.md

## What to build

Add 3D counterparts for the control preference critics that shape command behavior without defining the main geometric objective. The 3D planner should preserve the existing configuration style where practical while extending behavior to `[vx, vy, vz, wz]`.

This slice covers prefer-forward behavior, velocity deadband behavior, and twirling behavior. Twirling should remain focused on unnecessary yaw-rate use, not path yaw alignment.

## Acceptance criteria

- [x] Prefer-forward scoring works in the yaw-only 3D control space without introducing path-angle yaw alignment.
- [x] Velocity deadband scoring supports 4D controls.
- [x] Twirling scoring penalizes unnecessary yaw-rate behavior.
- [x] Configuration names and defaults stay close to the existing 2D critic conventions where practical.
- [x] Tests cover external scoring behavior for each 3D control preference critic.

## Blocked by

None - the 3D Core API slice is implemented and human-reviewed.

## Comments

- 2026-06-30: Implemented `PreferForwardCritic3D`, `VelocityDeadbandCritic3D`, and `TwirlingCritic3D`; wired them into `Optimizer3D`; exported the public 3D critic API; added focused behavior tests. Verification passed with `python3 -m pytest EXACT_MPPI_core/tests/test_mppi_3d_control_preference_critics.py`, `python3 -m pytest EXACT_MPPI_core/tests/test_mppi_3d_minimal_loop.py EXACT_MPPI_core/tests/test_mppi_3d_box_union_sdf.py EXACT_MPPI_core/tests/test_mppi_3d_obstacles.py EXACT_MPPI_core/tests/test_mppi_3d_goal_path_constraint_critics.py EXACT_MPPI_core/tests/test_mppi_3d_control_preference_critics.py`, and `python3 -m compileall -q EXACT_MPPI_core/exact_mppi/mppi_3d`.
