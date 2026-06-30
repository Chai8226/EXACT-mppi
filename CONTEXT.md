# EXACT-MPPI Core

This context describes the navigation and collision-checking language used by the core Python MPPI examples.

## Language

**Yaw-only 3D motion**:
Robot motion with 3D translation and vertical-axis yaw rotation. The state is `[x, y, z, yaw]`, and the control is `[vx, vy, vz, wz]`; roll and pitch are outside this motion model.
_Avoid_: Full 6DoF motion, SE(3) motion

**Yaw-only 3D holonomic motion model**:
A kinematic local-planning motion model where body-frame `vx` and `vy`, vertical `vz`, and yaw rate `wz` are directly controlled and integrated into yaw-only 3D motion.
_Avoid_: Dynamics model, physics simulation

**3D robot volume**:
The robot's three-dimensional body geometry used for collision evaluation against 3D obstacle points.
_Avoid_: Footprint, 2D footprint

**Box union volume**:
A 3D robot volume configured as the union of axis-aligned boxes in the robot body frame. The controller converts each box into closed triangle faces and halfspaces before optimization.
_Avoid_: Hand-written triangle soup, raw mesh config

**Convex polyhedron union**:
A 3D robot volume represented as the union of multiple convex polyhedra. Each convex part provides closed triangle faces for exact surface distance and halfspaces for inside/outside classification.
_Avoid_: Arbitrary mesh, halfspace clearance proxy

**Exact 3D polyhedron SDF**:
A signed distance calculation for a convex polyhedron that uses the minimum point-to-triangle surface distance for magnitude and halfspace classification for sign.
_Avoid_: Halfspace signed clearance, approximate SDF

**3D obstacle points**:
A fixed-size batch of sampled obstacle points in world coordinates with shape `(N, 3)`. Obstacle surfaces, maps, and meshes are represented only through these sampled points in the first 3D version.
_Avoid_: Obstacle mesh, occupancy grid, ESDF map

**Local 3D observation points**:
The obstacle point batch passed to the 3D controller after the example converts current observations into the robot-local yaw-only frame. This mirrors the existing Core Python examples where the controller receives local lidar points and a local plan.
_Avoid_: Global truth input, map-owned obstacle field

**Range-based local 3D observation**:
A Core-only observation model that selects nearby global 3D obstacle points within sensor range, keeps the nearest configured maximum, and transforms them into the robot-local yaw-only frame.
_Avoid_: 3D lidar raycast, full ir-sim 3D sensor

**3D reference path**:
A fixed-length local navigation path whose points are `[x, y, z, yaw]`. Position-based critics use 3D distance; yaw is constrained only by the goal yaw critic, not by a path-angle critic.
_Avoid_: 2D path, full orientation path

**Position-only 3D path alignment**:
Path alignment that penalizes distance from rollout points to a 3D reference path using only `x`, `y`, and `z`. Intermediate yaw is not aligned to the path.
_Avoid_: Path yaw alignment, PathAngleCritic3D
