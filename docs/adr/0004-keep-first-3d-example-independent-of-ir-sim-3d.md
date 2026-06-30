# Keep the first 3D example independent of ir-sim 3D

The first yaw-only 3D Core Python example will not depend on extending `ir-sim` into a full 3D simulator. The repository already contains partial 3D plotting/world scaffolding in `ir-sim_mppi`, but its object, sensor, geometry, and collision layers remain primarily 2D; the first 3D MPPI path should generate local 3D point observations directly and use matplotlib 3D visualization, while leaving a later `ir-sim` 3D phase explicitly out of scope.
