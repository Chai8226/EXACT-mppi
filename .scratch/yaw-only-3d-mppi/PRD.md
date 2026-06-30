# PRD: Yaw-Only 3D MPPI Core Python Path

Status: ready-for-agent

## Progress

- 2026-06-30: Issues 01 and 02 are implemented and have passing focused tests:
  `python3 -m pytest EXACT_MPPI_core/tests/test_mppi_3d_minimal_loop.py EXACT_MPPI_core/tests/test_mppi_3d_box_union_sdf.py`.
- 2026-06-30: Issue 03 is implemented and has passing focused tests:
  `python3 -m pytest EXACT_MPPI_core/tests/test_mppi_3d_minimal_loop.py EXACT_MPPI_core/tests/test_mppi_3d_box_union_sdf.py EXACT_MPPI_core/tests/test_mppi_3d_obstacles.py`.
- 2026-06-30: Issue 04 is implemented and has passing focused/full Core tests:
  `python3 -m pytest EXACT_MPPI_core/tests/test_mppi_3d_goal_path_constraint_critics.py`
  and `python3 -m pytest EXACT_MPPI_core/tests`.
- 2026-06-30: Issue 05 is implemented and has passing focused/3D Core tests:
  `python3 -m pytest EXACT_MPPI_core/tests/test_mppi_3d_control_preference_critics.py`
  and `python3 -m pytest EXACT_MPPI_core/tests/test_mppi_3d_minimal_loop.py EXACT_MPPI_core/tests/test_mppi_3d_box_union_sdf.py EXACT_MPPI_core/tests/test_mppi_3d_obstacles.py EXACT_MPPI_core/tests/test_mppi_3d_goal_path_constraint_critics.py EXACT_MPPI_core/tests/test_mppi_3d_control_preference_critics.py`.
- 2026-06-30: Issue 06 is implemented and has passing example/focused/full Core tests:
  `PYTHONPATH=EXACT_MPPI_core python3 EXACT_MPPI_core/example/yaw_only_3d_obstacle_avoidance/mppi_3d_obstacle_avoidance.py`,
  `python3 -m pytest EXACT_MPPI_core/tests/test_mppi_3d_obstacle_avoidance_example.py`,
  and `python3 -m pytest EXACT_MPPI_core/tests`.

## Problem Statement

The current Core Python MPPI implementation supports exact collision checking for 2D arbitrary robot shapes moving in `[x, y, yaw]`. It cannot plan for a robot that must move through three-dimensional space with state `[x, y, z, yaw]`, control `[vx, vy, vz, wz]`, a 3D robot volume, and 3D obstacle points.

The user wants a Core Python-only 3D version, not a ROS integration, that can actively change `x`, `y`, `z`, and `yaw` to avoid obstacles. The implementation should preserve the existing 2D package and examples while adding a parallel yaw-only 3D path with similar controller, critic, configuration, and visualization conventions.

## Solution

Add a parallel yaw-only 3D MPPI Core Python path. The new path should expose a 3D controller API, a yaw-only 3D holonomic motion model, exact 3D polyhedron SDF collision evaluation for box union volumes, 3D versions of the existing critic set except path-angle yaw alignment, and one Core-only example that demonstrates 3D obstacle avoidance.

The first 3D example should generate range-based local 3D observation points directly, run without ROS and without requiring `ir-sim` to become a full 3D simulator, and visualize the result with matplotlib 3D in a style similar to the existing examples.

## User Stories

1. As a researcher, I want a yaw-only 3D MPPI controller, so that I can plan in `[x, y, z, yaw]` instead of only `[x, y, yaw]`.
2. As a researcher, I want controls shaped as `[vx, vy, vz, wz]`, so that the planner can actively change altitude and yaw while navigating.
3. As a researcher, I want the existing 2D `mppi_jax` path left intact, so that current 2D examples and ROS-related workflows do not regress.
4. As a Core Python user, I want a 3D controller API similar to the current controller API, so that the 3D path feels familiar.
5. As a Core Python user, I want `computeVelocityCommands` to accept 3D local observation points, so that the controller can evaluate 3D obstacles.
6. As a Core Python user, I want `computeVelocityCommands` to return a finite 4D command, so that it can drive a yaw-only 3D simulation loop.
7. As a planner user, I want local plans and goals shaped as `[x, y, z, yaw]`, so that path and goal costs are defined in the same state space.
8. As a planner user, I want a yaw-only 3D holonomic motion model, so that `vx`, `vy`, `vz`, and `wz` are directly sampled and integrated.
9. As a planner user, I want the motion model to stay kinematic, so that the 3D implementation matches the abstraction level of the existing 2D planner.
10. As a planner user, I want no roll or pitch state, so that the first 3D version stays scoped to yaw-only 3D motion.
11. As a robot-shape author, I want to configure a 3D robot volume as a box union volume, so that I can express non-convex L/T-style bodies without hand-writing mesh data.
12. As a robot-shape author, I want each configured box converted internally to closed triangle faces and halfspaces, so that the public config stays readable while the SDF remains exact for the represented volume.
13. As a robot-shape author, I want non-convex robot volumes represented as convex polyhedron unions, so that complex bodies can be composed from simpler convex parts.
14. As a collision-checking user, I want exact 3D polyhedron SDF for configured convex parts, so that collision cost is based on point-to-surface distance rather than a halfspace clearance proxy.
15. As a collision-checking user, I want point-to-triangle distance used for SDF magnitude, so that distances to faces, edges, and vertices are handled correctly.
16. As a collision-checking user, I want halfspaces used for inside/outside classification, so that the SDF sign is stable for closed convex parts.
17. As a collision-checking user, I want union behavior across convex parts, so that the robot volume can be treated as one non-convex body.
18. As a planner user, I want obstacle input shaped as `(N, 3)`, so that the planner can consume 3D obstacle points.
19. As a planner user, I want obstacle masks to remain supported, so that fixed-size JAX arrays can represent variable-size observations.
20. As a Core-only example user, I want range-based local 3D observation points, so that the example mirrors the current local lidar point flow without needing a 3D lidar simulator.
21. As a Core-only example user, I want global obstacle points retained for visualization, so that I can inspect the planned path relative to the scenario.
22. As a Core-only example user, I want obstacle points transformed into the robot-local yaw-only frame before controller input, so that the controller sees local observations like existing examples do.
23. As a Core-only example user, I want a single 3D obstacle-avoidance example, so that I can quickly validate the new 3D planner.
24. As a Core-only example user, I want the example to reach a 3D goal, so that successful planning is obvious.
25. As a Core-only example user, I want the example to report minimum SDF clearance, so that collision avoidance can be checked numerically.
26. As a Core-only example user, I want the example to fail clearly if it collides or misses the goal, so that automated checks can use it.
27. As a visualization user, I want matplotlib 3D visualization, so that the example remains lightweight and close to the current Python examples.
28. As a visualization user, I want obstacle points, global reference path, local plan, rollouts, and optimal trajectory drawn with familiar colors, so that I can interpret the 3D run quickly.
29. As a visualization user, I want optional rollout visualization, so that expensive visual clutter can be enabled only when needed.
30. As a visualization user, I want optional GIF saving, so that I can inspect and share the 3D run.
31. As a headless test user, I want display/render toggles, so that CI or remote sessions can run the example without opening a GUI.
32. As a planner user, I want 3D counterparts for the existing critic set, so that the 3D planner has comparable behavior to the 2D planner.
33. As a planner user, I want no `PathAngleCritic3D`, so that yaw remains free for obstacle avoidance during the path.
34. As a planner user, I want goal yaw constrained only near the goal, so that yaw is not pulled toward the final orientation too early.
35. As a planner user, I want position-based critics to use `xyz` distance, so that vertical motion affects scoring correctly.
36. As a planner user, I want path alignment to be position-only, so that a 3D reference path guides position without constraining intermediate yaw.
37. As a planner user, I want path following to operate along the 3D reference path, so that progress is measured in 3D.
38. As a planner user, I want obstacle cost to rise near collision, so that trajectories keep clearance from 3D obstacle points.
39. As a planner user, I want velocity deadband behavior extended to 4D controls, so that unwanted near-zero commands can be discouraged consistently.
40. As a planner user, I want twirling behavior extended to yaw rate, so that unnecessary spinning can still be penalized.
41. As a maintainer, I want configuration names and semantics to stay close to the 2D package, so that the codebase remains navigable.
42. As a maintainer, I want tests at the highest useful seam, so that implementation details can change without breaking valid behavior.
43. As a maintainer, I want lower-level tests for the exact 3D polyhedron SDF, so that the riskiest math is protected.
44. As a maintainer, I want the first implementation to avoid full `ir-sim` 3D expansion, so that simulator work does not block the Core Python planner.
45. As a future simulator maintainer, I want the `ir-sim` 3D boundary documented, so that later work can extend objects, sensors, geometry, and collision deliberately.

## Implementation Decisions

- Add a parallel yaw-only 3D Core Python path rather than making the existing 2D package dimension-generic in place.
- Preserve the existing 2D controller, optimizer, models, critics, tools, examples, and ROS-facing behavior.
- Expose a 3D controller class whose public API mirrors the current controller style while making 3D concepts explicit.
- Use state `[x, y, z, yaw]` and speed/control `[vx, vy, vz, wz]`.
- Use a yaw-only 3D holonomic motion model:

```text
x_dot = vx * cos(yaw) - vy * sin(yaw)
y_dot = vx * sin(yaw) + vy * cos(yaw)
z_dot = vz
yaw_dot = wz
```

- Treat `vx` and `vy` as body-frame horizontal velocities, `vz` as vertical velocity, and `wz` as yaw rate.
- Keep the motion model kinematic; do not add mass, inertia, gravity, thrust, contact, roll, or pitch dynamics.
- Use 3D reference path points and goals shaped as `[x, y, z, yaw]`.
- Use local 3D observation points shaped as `(N, 3)` with a mask shaped as `(N,)`.
- Rename the 3D obstacle input to `obstacle_points`; do not call it `lidar_points` in the 3D API.
- Configure the first 3D robot volume as a box union volume.
- Convert each configured box into closed triangle faces and halfspaces internally.
- Use point-to-triangle distance for exact 3D polyhedron SDF magnitude.
- Use halfspace classification for exact 3D polyhedron SDF sign.
- Use minimum SDF across convex parts to represent a convex polyhedron union.
- Port the existing critic set to 3D except `PathAngleCritic3D`.
- Keep `ConstraintCritic3D`, `GoalCritic3D`, `GoalYawCritic3D`, `ObstaclesCritic3D`, `PathAlignCritic3D`, `PathFollowCritic3D`, `PreferForwardCritic3D`, `VelocityDeadbandCritic3D`, and `TwirlingCritic3D`.
- Make position-based critic distances operate on `x`, `y`, and `z`.
- Make yaw-specific behavior compare only yaw.
- Make goal yaw scoring active only when the 3D position is within its configured threshold.
- Make path alignment position-only; it should not align intermediate yaw to the path.
- Do not add path-angle yaw alignment in the 3D path.
- Build one Core Python 3D example that demonstrates 3D obstacle avoidance.
- The example should keep the current examples' local-frame loop shape: global scenario data for visualization, local plan and local obstacle points for controller input, and controller pose set to the local origin.
- The example should use range-based local 3D observation points instead of a full 3D lidar raycast.
- The example should use matplotlib 3D visualization, including optional rollout display and optional GIF saving.
- Leave full `ir-sim` 3D simulator expansion out of scope for this PRD.

## Testing Decisions

- Prefer tests at the highest useful seam: the Core-only 3D example should validate the feature end to end by reaching the 3D goal while maintaining minimum exact 3D polyhedron SDF above the collision margin.
- Add focused tests for the exact 3D polyhedron SDF because it is the highest-risk mathematical surface.
- SDF tests should cover point-to-triangle distance, inside/outside halfspace classification, signed distance for boxes, and union behavior for box union volumes.
- Add a public API test for the 3D controller seam: `computeVelocityCommands` should accept `[x, y, z, yaw]`, `[vx, vy, vz, wz]`, a 3D reference path, a 3D goal, and local 3D observation points, then return a finite 4D command.
- Add critic behavior tests that assert external behavior, not implementation details.
- Critic tests should verify that obstacle cost rises near collision, goal/path costs use `xyz`, goal yaw only applies near the goal, and no path-angle yaw alignment is present.
- Existing 2D example and controller behavior should remain covered by existing tests or smoke checks where available; the 3D work should not require changing their public behavior.
- Use small deterministic scenarios for tests so they run on CPU and do not require ROS, Gazebo, or a GPU.

## Out of Scope

- Full 6DoF motion.
- Roll and pitch state or control.
- Physics simulation, gravity, thrust, mass, inertia, contact, suspension, or aerodynamic modeling.
- ROS integration for the 3D path.
- Gazebo or RViz integration for the 3D path.
- Full `ir-sim` 3D simulator expansion.
- 3D lidar raycasting or realistic sensor simulation.
- Obstacle meshes, occupancy grids, OctoMap, ESDF maps, or map-owned obstacle fields.
- Arbitrary triangle mesh robot volume as the first public shape format.
- A `PathAngleCritic3D`.
- Multiple 3D examples; the first PRD requires one example that demonstrates 3D obstacle avoidance.
- Refactoring the existing 2D package into a dimension-generic framework.

## Further Notes

- This PRD follows the glossary in `CONTEXT.md`.
- This PRD respects the ADRs for closed convex polyhedron unions, a parallel 3D Core Python path, no path-angle yaw alignment, keeping the first 3D example independent of `ir-sim` 3D, and using box-union YAML for first 3D volume config.
- The current `ir-sim_mppi` fork already contains partial 3D world and plotting scaffolding, but its object, sensor, geometry, and collision layers remain primarily 2D. That can become a later phase after the Core Python 3D path is working.
