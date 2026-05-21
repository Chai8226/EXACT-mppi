# Docker Guide

This repository can be developed and tested inside Docker. The recommended setup is:

- Ubuntu 22.04
- ROS 2 Humble
- NVIDIA CUDA 12
- Python virtual environment inside the repository

The repository already includes a base image definition at `docker/Dockerfile.humble-cuda`.

## What This Container Is For

Use the Docker environment when you want:

- a reproducible Ubuntu 22.04 + ROS 2 Humble workspace
- GPU support for JAX
- Gazebo and RViz-based ROS 2 demos without polluting the host system

The provided Dockerfile is a good base for development. For ROS 2 simulation workflows in `mosaic_mppi_ros2`, you should additionally install Gazebo, `gazebo_ros`, `xacro`, and `rviz2` inside the image.

## Prerequisites

Before building the container, make sure the host machine has:

- Docker
- NVIDIA Container Toolkit
- a working NVIDIA driver
- X11 available if you want to run Gazebo or RViz with GUI

Quick checks on the host:

```bash
docker --version
nvidia-smi
echo "$DISPLAY"
ls "$XSOCK"
```

If `nvidia-smi` fails on the host, fix the GPU driver stack before debugging the container.

## Build the Image

From the repository root:

```bash
cd EXACT-mppi
docker build -f docker/Dockerfile.humble-cuda -t exact-mppi:humble-cuda .
```

## Run the Container

Allow the local root user in the container to access your X server:

```bash
xhost +local:root
```

Set two shell variables on the host before starting the container:

```bash
export XSOCK=...
export CONTAINER_WS=...
```

`XSOCK` should point to the host X11 socket directory, and `CONTAINER_WS` should be the workspace path you want to use inside the container.

Start the container with GPU and X11 forwarding:

```bash
docker run -it --rm \
  --gpus all \
  --network host \
  --ipc host \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v "$XSOCK:$XSOCK:rw" \
  -v "$(pwd):$CONTAINER_WS" \
  -w "$CONTAINER_WS" \
  --name exact-mppi-dev \
  exact-mppi:humble-cuda
```

This command mounts the repository into the container and drops you into the project workspace.

## Install Additional ROS 2 Simulation Tools

The base Dockerfile currently installs `ros-humble-ros-base`, which is enough for headless ROS 2 development but not enough for the Gazebo and RViz launch files used in `mosaic_mppi_ros2`.

Inside the container, install the missing simulation packages:

```bash
apt update
apt install -y \
  gazebo \
  ros-humble-gazebo-ros-pkgs \
  ros-humble-gazebo-plugins \
  ros-humble-gazebo-msgs \
  ros-humble-xacro \
  ros-humble-rviz2 \
  ros-humble-joint-state-publisher \
  ros-humble-robot-state-publisher
```

If you want these packages to be available every time you build the image, add the same package list to `docker/Dockerfile.humble-cuda`.

## Create the Python Environment

Inside the container:

```bash
cd .
python3 -m venv .exact_mppi
source .exact_mppi/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Install the local Python packages:

```bash
python -m pip install -e ./ir-sim_mppi
python -m pip install -e ./EXACT_MPPI_core
```

For GPU-enabled JAX on CUDA 12:

```bash
python -m pip install -U "jax[cuda12]"
```

If your environment targets a different CUDA major version, choose the matching JAX build instead.

## Build the ROS 2 Workspace

The ROS 2 bridge workspace is located in `mosaic_mppi_ros2`.

Inside the container:

```bash
cd mosaic_mppi_ros2
./setup.sh
./build.sh
source install/setup.bash
```

If `setup.sh` is not executable:

```bash
chmod +x setup.sh build.sh
```

## Verification

Check that ROS 2 is available:

```bash
source ~/.bashrc
ros2 --version
```

Check that JAX can see the GPU:

```bash
source .exact_mppi/bin/activate
python -c "import jax; print(jax.devices())"
```

Check that RViz is installed:

```bash
source ~/.bashrc
rviz2
```

## Run a First Example

Core Python example:

```bash
source .exact_mppi/bin/activate
python EXACT_MPPI_core/example/corridor_dynamic_random_jax/mppi_jax_anyshape.py --robot-shape f
```

ROS 2 simulation example:

```bash
cd mosaic_mppi_ros2
source install/setup.bash
ros2 launch exact_mppi_jax sim_limo_corridor_launch.py auto_goal:=true
```

## Common Issues

`rviz2: command not found`

Install:

```bash
apt install -y ros-humble-rviz2
```

`package 'gazebo_ros' not found`

Install:

```bash
apt install -y ros-humble-gazebo-ros-pkgs
```

`file not found: xacro`

Install:

```bash
apt install -y ros-humble-xacro
```

Gazebo or RViz cannot open a window

Check:

- `echo $DISPLAY` is not empty
- the X11 socket directory is mounted into the container
- `xhost +local:root` was run on the host

JAX does not use GPU

Check:

- `nvidia-smi` works on the host
- the container was started with `--gpus all`
- the installed JAX package matches the CUDA major version

## Recommended Next Step

If you want a one-command setup for new users, the next cleanup step is to move the simulation packages from the manual `apt install` section into `docker/Dockerfile.humble-cuda`, so the image is ready for `mosaic_mppi_ros2` immediately after `docker build`.
