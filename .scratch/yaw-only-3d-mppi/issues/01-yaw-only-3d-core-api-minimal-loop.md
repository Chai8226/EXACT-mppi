# 建立 yaw-only 3D Core API 的最小闭环

Status: ready-for-human

## Parent

.scratch/yaw-only-3d-mppi/PRD.md

## What to build

Add the first usable yaw-only 3D Core Python path as a parallel API that does not change the existing 2D planner. This slice should make the 3D controller, state/control models, yaw-only 3D holonomic motion model, optimizer path, and public command API coherent enough to run a no-obstacle smoke scenario and return a finite `[vx, vy, vz, wz]` command.

The goal is not to finish collision checking or the full critic set yet. The goal is to establish the 3D package shape and prove that `[x, y, z, yaw]` planning can execute through the public API without disturbing the 2D path.

## Acceptance criteria

- [x] A parallel 3D Core Python API exists and can be imported without changing the existing 2D imports.
- [x] The 3D controller accepts state `[x, y, z, yaw]`, speed `[vx, vy, vz, wz]`, a 3D reference path, a 3D goal, and local 3D observation points.
- [x] The 3D yaw-only holonomic motion model integrates body-frame `vx`, `vy`, vertical `vz`, and yaw rate `wz` into candidate trajectories.
- [x] A no-obstacle smoke test or equivalent executable check returns a finite 4D command.
- [x] Existing 2D examples and public controller imports remain unchanged.

## Blocked by

None - can start immediately

## Comments

- 2026-06-30: Verified against `EXACT_MPPI_core/exact_mppi/mppi_3d/` and `EXACT_MPPI_core/tests/test_mppi_3d_minimal_loop.py`. `python3 -m pytest EXACT_MPPI_core/tests/test_mppi_3d_minimal_loop.py EXACT_MPPI_core/tests/test_mppi_3d_box_union_sdf.py` passed with 8 tests.
