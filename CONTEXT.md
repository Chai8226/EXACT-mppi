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
The robot's three-dimensional body geometry used for collision evaluation against observed 3D obstacle points.
_Avoid_: Footprint, 2D footprint

**Box union volume**:
A 3D robot volume configured as the union of axis-aligned boxes in the robot body frame. The controller converts each box into closed triangle faces and halfspaces before optimization.
_Avoid_: Hand-written triangle soup, raw mesh config

**T-shaped 3D robot volume**:
A non-convex 3D robot volume composed from body-frame boxes in a T layout and used as the authoritative collision geometry for yaw-only 3D navigation.
_Avoid_: Display-only T model, decorative drone model

**Convex polyhedron union**:
A 3D robot volume represented as the union of multiple convex polyhedra. Each convex part provides closed triangle faces for exact surface distance and halfspaces for inside/outside classification.
_Avoid_: Arbitrary mesh, halfspace clearance proxy

**Exact 3D polyhedron SDF**:
A signed distance calculation for a convex polyhedron that uses the minimum point-to-triangle surface distance for magnitude and halfspace classification for sign.
_Avoid_: Halfspace signed clearance, approximate SDF

**Legacy 3D obstacle points**:
A pre-raycast obstacle representation where obstacle surfaces are approximated by sampled world-frame points. This belongs to the first 3D version and is not the authority for MID-360-like observation.
_Avoid_: Authoritative obstacle geometry, observed point cloud

**3D scenario suite**:
A set of reproducible yaw-only 3D navigation scenarios where each scenario isolates one capability such as baseline tracking, vertical gates, narrow gaps, T-shaped traps, or cluttered corridors.
_Avoid_: One-off demo world, single showcase scene

**3D scenario definition**:
A normalized in-memory description of one yaw-only 3D scenario: simulation settings, 3D reference path, 3D robot volume, authoritative obstacle geometry, MID-360-like sensor settings, and controller point budget.
_Avoid_: Raw YAML mapping, partially-normalized runner config

**Local 3D observation points**:
The obstacle point batch passed to the 3D controller after the example converts current observations into the robot-local yaw-only frame. This mirrors the existing Core Python examples where the controller receives local lidar points and a local plan.
_Avoid_: Global truth input, map-owned obstacle field

**Range-based local 3D observation**:
A Core-only observation model that selects nearby global 3D obstacle points within sensor range, keeps the nearest configured maximum, and transforms them into the robot-local yaw-only frame.
_Avoid_: 3D lidar raycast, full ir-sim 3D sensor

**MID-360-like 3D point cloud observation**:
A simulated local 3D obstacle observation shaped by a Livox MID-360 class sensor envelope: omnidirectional horizontal coverage, asymmetric vertical coverage, a short blind range, and a configured conservative maximum range.
_Avoid_: Perfect local obstacle oracle, generic spherical range crop

**Observed 3D point cloud**:
The world-frame points produced for a simulation step by raycasting a MID-360-like sensor model against obstacle geometry. It is the complete simulated sensor output before controller point-budget selection.
_Avoid_: Global obstacle points, offline obstacle samples

**3D geometry observation**:
The Core simulation step's use of authoritative obstacle geometry and a yaw-only robot pose to produce the observed 3D point cloud, the robot-local observation given to the controller, and world-geometry clearance for scenario metrics.
_Avoid_: Controller-side obstacle cost, global obstacle points, browser replay rendering

**Controller 3D obstacle points**:
The robot-local yaw-frame subset of the observed 3D point cloud passed to the 3D controller under its fixed obstacle point budget. When the observed cloud exceeds the budget, the subset follows the original 2D controller behavior by keeping the nearest points by local range and padding the remainder with an invalid mask.
_Avoid_: Global obstacle points, range-cropped map truth

**Live 3D MPPI visualization**:
The Core-only yaw-only 3D example's interactive matplotlib view, updated during each planning step. It mirrors the 2D examples' run-and-watch workflow while keeping headless and GIF-only modes available for tests and remote runs.
_Avoid_: Post-run-only visualization, offline replay as the default example experience

**Passive Web 3D viewer**:
A browser-based 3D visualization backend that consumes frames emitted by the Python simulation loop without owning MPPI control, state integration, or scenario progression.
_Avoid_: Browser-owned simulator, Web control loop

**Offline Web replay**:
A passive Web 3D viewer mode where Python first completes a deterministic simulation run and exports replay data for browser-side loading, scrubbing, and inspection.
_Avoid_: Real-time Web simulator, live streaming as the first viewer mode

**3D smoothness telemetry**:
Per-step control and state-history measurements used to judge whether yaw-only 3D navigation is smooth independently of viewer playback interpolation.
_Avoid_: Animation smoothness, frame-rate smoothness

**3D replay baseline**:
A reproducible scenario-suite run that exports metrics and offline Web replay data before MPPI parameter or cost-function tuning begins.
_Avoid_: Visual-only tuning run, one-off parameter tweak

**2D-compatible 3D workflow**:
A yaw-only 3D example workflow that mirrors the 2D Core Python loop shape, configuration vocabulary, and controller-facing data flow while keeping the 3D optimizer implementation separate.
_Avoid_: Dimension-generic optimizer refactor, shared 2D/3D optimizer core

**3D reference path**:
A fixed-length local navigation path whose points are `[x, y, z, yaw]`. Position-based critics use 3D distance; yaw is constrained only by the goal yaw critic, not by a path-angle critic.
_Avoid_: 2D path, full orientation path

**Position-only 3D path alignment**:
Path alignment that penalizes distance from rollout points to a 3D reference path using only `x`, `y`, and `z`. Intermediate yaw is not aligned to the path.
_Avoid_: Path yaw alignment, PathAngleCritic3D
