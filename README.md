# Anonymous Code Release

Anonymous code release for a pre-print submission. This repository contains two local Python packages:

- `ir-sim_mppi`: simulator package used by the examples
- `EXACT_MPPI_core`: MPPI controller package and example scripts

## First-time setup

GPU execution is recommended for the JAX-based controller examples in `EXACT_MPPI_core/example/`. CPU execution works for setup checks and small tests, but the main MPPI examples are noticeably slower.

Use Python 3.10+ (3.10 recommended). Create the virtual environment directly inside this repository:

```bash
python3 -m venv .exact_mppi
source .exact_mppi/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

If `python3` is not Python 3.10+, replace it with a suitable interpreter such as `python3.10`. Python 3.10 is recommended for the most predictable package compatibility.

## Install the two local packages

From the repository root:

```bash
python -m pip install -e ./ir-sim_mppi
python -m pip install -e ./EXACT_MPPI_core
```

The editable `exact_mppi` install brings in the default JAX-based controller dependencies. The legacy torch-based modules are not installed by default.

If you also need the legacy torch-based MPPI / DUNE components, install the optional dependencies explicitly:

```bash
python -m pip install torch arm-pytorch-utilities
```

## Install JAX

Most examples in `EXACT_MPPI_core/example/` use JAX.

GPU is the suggested default when you have an NVIDIA GPU with a supported CUDA stack.

Recommended order:

```bash
python -m pip install -e ./ir-sim_mppi
python -m pip install -U "jax[cuda12]"
python -m pip install -e ./EXACT_MPPI_core
```

Replace `jax[cuda12]` with the CUDA family that matches your machine.

For CPU-only execution:

```bash
python -m pip install "jax[cpu]"
```

For NVIDIA GPU execution, the JAX build must match both your Python version and your installed CUDA version.

Check your versions first:

```bash
python --version
nvidia-smi
nvcc --version
```

Use a JAX wheel that matches your installed CUDA major version. If CUDA is not installed, or if you are unsure about compatibility, use the CPU-only build above.

Examples:

```bash
# If nvcc reports CUDA 12.9, install the CUDA 12 JAX build
python -m pip install -U "jax[cuda12]"

# If nvcc reports CUDA 13.0, install the CUDA 13 JAX build
python -m pip install -U "jax[cuda13]"
```

Match the JAX package to the CUDA major version only. For example, CUDA 12.9 still uses `jax[cuda12]`. 

For CUDA-enabled JAX, follow the official JAX installation instructions for the exact Python and CUDA combination on your machine:

https://docs.jax.dev/en/latest/installation.html

## Quick check

```bash
python -c "import irsim; from exact_mppi.mppi_jax.controller import MPPIController; print('setup ok')"
```

## Run a first example

```bash
source .exact_mppi/bin/activate
python EXACT_MPPI_core/example/corridor_dynamic_random_jax/mppi_jax_anyshape.py --robot-shape f
```

Each core example now prints the detected JAX backend and device list before the simulation starts, so you can confirm whether it is running on CPU or GPU.

For other scenarios, run a different script under `EXACT_MPPI_core/example/`.