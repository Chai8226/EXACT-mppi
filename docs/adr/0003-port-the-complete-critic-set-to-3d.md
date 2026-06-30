# Port the critic set to 3D except path-angle yaw alignment

The yaw-only 3D Core Python path will include 3D counterparts for the existing 2D critic set rather than a minimal proof-of-concept critic subset, except it will not include `PathAngleCritic3D`. Configuration names and critic semantics should stay close to the 2D package, with path and goal points shaped as `[x, y, z, yaw]`, position distances computed in 3D, and yaw constrained only near the goal so yaw remains available as an obstacle-avoidance degree of freedom during navigation.
