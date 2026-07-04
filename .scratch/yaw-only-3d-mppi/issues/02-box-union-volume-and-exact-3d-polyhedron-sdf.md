# 实现 box union volume 与 exact 3D polyhedron SDF

Status: implemented

## Parent

.scratch/yaw-only-3d-mppi/PRD.md

## What to build

Implement the 3D robot volume representation and exact 3D polyhedron SDF needed by the yaw-only 3D planner. The first public volume format is a box union volume: users configure body-frame boxes, and the system converts them into closed triangle faces and halfspaces.

The SDF should compute magnitude from exact point-to-triangle surface distance and sign from halfspace inside/outside classification. Union behavior should represent non-convex robot volumes by taking the minimum signed distance across convex parts.

## Acceptance criteria

- [x] A box union volume can be created from readable box configuration data.
- [x] Each configured box is converted into closed triangle faces and halfspaces internally.
- [x] Point-to-triangle distance handles face, edge, and vertex closest-point cases.
- [x] Signed distance for a single box is negative inside, near zero on the surface, and positive outside.
- [x] Union signed distance works across multiple boxes.
- [x] CPU-friendly tests cover point-to-triangle distance, halfspace classification, single-box SDF, and box-union SDF.

## Blocked by

None - the 3D Core API slice is implemented and human-reviewed.

## Comments

- 2026-06-30: Verified against `EXACT_MPPI_core/exact_mppi/mppi_3d/geometry.py` and `EXACT_MPPI_core/tests/test_mppi_3d_box_union_sdf.py`. `python3 -m pytest EXACT_MPPI_core/tests/test_mppi_3d_minimal_loop.py EXACT_MPPI_core/tests/test_mppi_3d_box_union_sdf.py` passed with 8 tests.
