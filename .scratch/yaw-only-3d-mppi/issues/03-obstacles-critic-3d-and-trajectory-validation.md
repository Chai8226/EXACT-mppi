# 接入 ObstaclesCritic3D 与 trajectory validation

Status: implemented

## Parent

.scratch/yaw-only-3d-mppi/PRD.md

## What to build

Connect exact 3D polyhedron SDF collision evaluation into the yaw-only 3D planner through obstacle scoring and optimal trajectory validation. The planner should consume local 3D observation points and masks, evaluate obstacle clearance over generated trajectories, penalize collisions and near-collisions, and report whether the selected trajectory remains above the collision margin.

This slice should make a simple obstacle-aware 3D control call prefer a safer trajectory over a colliding one.

## Acceptance criteria

- [x] The 3D obstacle critic evaluates local 3D observation points against the 3D robot volume over candidate trajectories.
- [x] Collision and near-collision costs use exact 3D polyhedron SDF values.
- [x] Invalid or padded observation points are ignored through the observation mask.
- [x] The 3D trajectory validator checks the optimal trajectory against the configured collision margin.
- [x] A deterministic test or smoke scenario shows obstacle cost increases near collision.
- [x] A deterministic test or smoke scenario reports minimum clearance for the selected trajectory.

## Blocked by

None - the 3D Core API and box-union SDF slices are implemented and human-reviewed.

## Comments

- 2026-06-30: Implemented `ObstaclesCritic3D` scoring, exact 3D clearance helpers, `OptimalTrajectoryValidator3D`, optimizer/controller wiring, and public getters for validation result and minimum clearance. Verified with `python3 -m pytest EXACT_MPPI_core/tests/test_mppi_3d_minimal_loop.py EXACT_MPPI_core/tests/test_mppi_3d_box_union_sdf.py EXACT_MPPI_core/tests/test_mppi_3d_obstacles.py`, which passed with 11 tests.
