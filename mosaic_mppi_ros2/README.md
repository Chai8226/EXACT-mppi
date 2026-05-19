# EXACT MPPI ROS2 Workspace

This workspace contains the ROS2 bridge around the installed `exact_mppi` Python package and a lightweight simulator used for local testing.

ROS 2 Humble is the suggested distro for this workspace.

Package layout:

- `src/exact_mppi_jax`: ROS2 bridge package that keeps the existing ROS interface while running the `exact_mppi` controller.
- `src/ddr_minimal_sim`: lightweight differential-drive simulator used by the example launches.

## Quick Start

Install ROS dependencies:

```bash
./setup.sh
```

Build the workspace:

```bash
./build.sh
```

Launch the corridor simulator demo:

```bash
source install/setup.bash
ros2 launch exact_mppi_jax sim_corridor_external_ref_launch.py
```

Launch the T-shape simulator demo:

```bash
source install/setup.bash
ros2 launch exact_mppi_jax sim_tshape_external_ref_launch.py
```

## Notes

- The ROS package name is `exact_mppi_jax`.
- The controller implementation itself is provided by the installed `exact_mppi` Python package.
- RViz configuration is included under `src/exact_mppi_jax/rviz/mppi_sim.rviz`.
