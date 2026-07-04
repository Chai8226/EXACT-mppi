# 构建 Core-only 3D obstacle-avoidance example

Status: implemented

## Parent

.scratch/yaw-only-3d-mppi/PRD.md

## What to build

Build the first Core-only yaw-only 3D obstacle-avoidance example. The example should generate a global 3D obstacle point scenario, produce range-based local 3D observation points each control cycle, transform the local plan and local observation points into the robot-local yaw-only frame, call the 3D controller at the local origin, and step the simulated yaw-only 3D state with the returned `[vx, vy, vz, wz]` command.

The example should be runnable without ROS, Gazebo, GPU, or a full `ir-sim` 3D simulator. It should report whether it reached the 3D goal and what minimum exact 3D polyhedron SDF clearance was observed.

## Acceptance criteria

- [x] The example runs as a Core Python script without ROS or Gazebo.
- [x] The example generates or loads global 3D obstacle points for scenario and visualization use.
- [x] Each control cycle builds range-based local 3D observation points and a mask.
- [x] The controller receives local 3D observation points, a local 3D plan, a local 3D goal, and local-origin robot pose.
- [x] The simulated state updates according to the yaw-only 3D holonomic model.
- [x] A deterministic headless run reaches the 3D goal.
- [x] The run reports minimum SDF clearance and fails clearly if it collides or misses the goal.

## Implementation notes

- Added `EXACT_MPPI_core/example/yaw_only_3d_obstacle_avoidance/mppi_3d_obstacle_avoidance.py`.
- Added headless tests in `EXACT_MPPI_core/tests/test_mppi_3d_obstacle_avoidance_example.py`.
- Verified with:
  - `PYTHONPATH=EXACT_MPPI_core python3 EXACT_MPPI_core/example/yaw_only_3d_obstacle_avoidance/mppi_3d_obstacle_avoidance.py`
  - `python3 -m pytest EXACT_MPPI_core/tests/test_mppi_3d_obstacle_avoidance_example.py -q`
  - `python3 -m pytest EXACT_MPPI_core/tests -q`

## Blocked by

- .scratch/yaw-only-3d-mppi/issues/04-3d-goal-path-constraint-critics.md
- .scratch/yaw-only-3d-mppi/issues/05-3d-control-preference-critics.md
