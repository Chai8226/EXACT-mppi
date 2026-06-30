# 构建 Core-only 3D obstacle-avoidance example

Status: ready-for-agent

## Parent

.scratch/yaw-only-3d-mppi/PRD.md

## What to build

Build the first Core-only yaw-only 3D obstacle-avoidance example. The example should generate a global 3D obstacle point scenario, produce range-based local 3D observation points each control cycle, transform the local plan and local observation points into the robot-local yaw-only frame, call the 3D controller at the local origin, and step the simulated yaw-only 3D state with the returned `[vx, vy, vz, wz]` command.

The example should be runnable without ROS, Gazebo, GPU, or a full `ir-sim` 3D simulator. It should report whether it reached the 3D goal and what minimum exact 3D polyhedron SDF clearance was observed.

## Acceptance criteria

- [ ] The example runs as a Core Python script without ROS or Gazebo.
- [ ] The example generates or loads global 3D obstacle points for scenario and visualization use.
- [ ] Each control cycle builds range-based local 3D observation points and a mask.
- [ ] The controller receives local 3D observation points, a local 3D plan, a local 3D goal, and local-origin robot pose.
- [ ] The simulated state updates according to the yaw-only 3D holonomic model.
- [ ] A deterministic headless run reaches the 3D goal.
- [ ] The run reports minimum SDF clearance and fails clearly if it collides or misses the goal.

## Blocked by

- .scratch/yaw-only-3d-mppi/issues/04-3d-goal-path-constraint-critics.md
- .scratch/yaw-only-3d-mppi/issues/05-3d-control-preference-critics.md
