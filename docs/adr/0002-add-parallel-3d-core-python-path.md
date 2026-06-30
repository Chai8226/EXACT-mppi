# Add a parallel 3D Core Python path

The yaw-only 3D implementation will be added as a parallel Core Python path rather than by making the existing `mppi_jax` package dimension-generic in place. The current 2D package and examples are heavily shaped around `[x, y, yaw]`, `[vx, vy, wz]`, and `(N, 2)` obstacle points; a parallel 3D path keeps those examples stable while allowing the 3D API to use `[x, y, z, yaw]`, `[vx, vy, vz, wz]`, and `(N, 3)` obstacle points explicitly.
