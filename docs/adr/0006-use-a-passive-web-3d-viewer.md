# Use a passive Web 3D viewer for the next 3D visualization

The next 3D visualization will use a browser-based viewer as a passive rendering and interaction surface while Python keeps ownership of the MPPI control loop, state integration, local observations, local plan selection, and scenario progression. This keeps the yaw-only 3D navigation flow close to the original 2D Core Python examples, while still allowing a more modern and inspectable 3D interface than the first matplotlib view.
