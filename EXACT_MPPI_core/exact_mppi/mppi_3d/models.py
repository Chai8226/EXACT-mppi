from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import tree_util


@tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class ControlConstraints3D:
    """Limits for yaw-only 3D controls."""

    vx_max: float
    vx_min: float
    vy: float
    vz: float
    wz: float


@tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class SamplingStd3D:
    """Noise standard deviations for 4D control sampling."""

    vx: float
    vy: float
    vz: float
    wz: float


@tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class ControlSequence3D:
    """A yaw-only 3D control sequence over time."""

    vx: jax.Array  # (T,)
    vy: jax.Array  # (T,)
    vz: jax.Array  # (T,)
    wz: jax.Array  # (T,)


@tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class Trajectories3D:
    """Candidate yaw-only 3D trajectories."""

    x: jax.Array  # (K, T)
    y: jax.Array  # (K, T)
    z: jax.Array  # (K, T)
    yaws: jax.Array  # (K, T)


@tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class OptimizerSettings3D:
    """Settings for the minimal yaw-only 3D MPPI optimizer."""

    constraints: ControlConstraints3D
    sampling_std: SamplingStd3D
    model_dt: float
    temperature: float
    batch_size: int
    time_steps: int
    iteration_count: int
    shift_control_sequence: bool
    goal_weight: float
    goal_yaw_weight: float
    path_weight: float
    control_weight: float


def reset_ControlSequence3D(
    time_steps: int, dtype: jnp.dtype = jnp.float32
) -> ControlSequence3D:
    return ControlSequence3D(
        vx=jnp.zeros(time_steps, dtype=dtype),
        vy=jnp.zeros(time_steps, dtype=dtype),
        vz=jnp.zeros(time_steps, dtype=dtype),
        wz=jnp.zeros(time_steps, dtype=dtype),
    )
