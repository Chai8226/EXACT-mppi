from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import tree_util


@tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class PreferForwardCriticParams3D:
    """Forward-motion preference parameters for yaw-only 3D controls."""

    enabled: bool
    power: int
    weight: float
    threshold_to_consider: float


@tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class VelocityDeadbandCriticParams3D:
    """Velocity deadband parameters for 4D yaw-only 3D controls."""

    enabled: bool
    power: int
    weight: float
    deadband_velocities: jax.Array


@tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class TwirlingCriticParams3D:
    """Yaw-rate twirling parameters for yaw-only 3D controls."""

    enabled: bool
    power: int
    weight: float
    pose_tolerance: float


def prefer_forward_critic_initialize_3d(
    critic_params_dict: dict,
) -> PreferForwardCriticParams3D:
    return PreferForwardCriticParams3D(
        enabled=critic_params_dict.get("enabled", True),
        power=critic_params_dict.get("cost_power", 1),
        weight=critic_params_dict.get("cost_weight", 5.0),
        threshold_to_consider=critic_params_dict.get("threshold_to_consider", 0.5),
    )


def velocity_deadband_critic_initialize_3d(
    critic_params_dict: dict,
) -> VelocityDeadbandCriticParams3D:
    return VelocityDeadbandCriticParams3D(
        enabled=critic_params_dict.get("enabled", True),
        power=critic_params_dict.get("cost_power", 1),
        weight=critic_params_dict.get("cost_weight", 35.0),
        deadband_velocities=jnp.asarray(
            critic_params_dict.get(
                "deadband_velocities",
                jnp.array([0.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
            ),
            dtype=jnp.float32,
        ),
    )


def twirling_critic_initialize_3d(
    critic_params_dict: dict,
) -> TwirlingCriticParams3D:
    return TwirlingCriticParams3D(
        enabled=critic_params_dict.get("enabled", True),
        power=critic_params_dict.get("cost_power", 1),
        weight=critic_params_dict.get("cost_weight", 10.0),
        pose_tolerance=critic_params_dict.get("pose_tolerance", 0.1),
    )


def prefer_forward_critic_score_3d(
    vx: jax.Array,
    local_path_length: jax.Array,
    params: PreferForwardCriticParams3D,
    model_dt: float,
) -> Tuple[jax.Array, dict]:
    """Score reverse body-frame x velocity when there is enough path to follow."""

    batch_size = vx.shape[0]
    if not params.enabled:
        return jnp.zeros((batch_size,), dtype=jnp.float32), {}

    should_skip = jnp.asarray(local_path_length) < params.threshold_to_consider

    def skip_score(_):
        return jnp.zeros((batch_size,), dtype=jnp.float32), {}

    def do_score(_):
        cost = jnp.sum(jax.nn.relu(-vx) * model_dt, axis=1) * params.weight
        cost = _apply_power(cost, params.power)
        return cost, {}

    return jax.lax.cond(should_skip, skip_score, do_score, operand=None)


def velocity_deadband_critic_score_3d(
    vx: jax.Array,
    vy: jax.Array,
    vz: jax.Array,
    wz: jax.Array,
    params: VelocityDeadbandCriticParams3D,
    model_dt: float,
) -> Tuple[jax.Array, dict]:
    """Score controls inside configured 4D velocity deadbands."""

    batch_size = vx.shape[0]
    if not params.enabled:
        return jnp.zeros((batch_size,), dtype=jnp.float32), {}

    deadband = params.deadband_velocities
    cost = (
        (
            jax.nn.relu(jnp.abs(deadband[0]) - jnp.abs(vx))
            + jax.nn.relu(jnp.abs(deadband[1]) - jnp.abs(vy))
            + jax.nn.relu(jnp.abs(deadband[2]) - jnp.abs(vz))
            + jax.nn.relu(jnp.abs(deadband[3]) - jnp.abs(wz))
        )
        * model_dt
    ).sum(axis=1) * params.weight
    cost = _apply_power(cost, params.power)
    return cost, {}


def twirling_critic_score_3d(
    wz: jax.Array,
    local_path_length: jax.Array,
    params: TwirlingCriticParams3D,
) -> Tuple[jax.Array, dict]:
    """Score unnecessary yaw-rate use when there is enough path to follow."""

    batch_size = wz.shape[0]
    if not params.enabled:
        return jnp.zeros((batch_size,), dtype=jnp.float32), {}

    should_skip = jnp.asarray(local_path_length) < params.pose_tolerance

    def skip_score(_):
        return jnp.zeros((batch_size,), dtype=jnp.float32), {}

    def do_score(_):
        cost = jnp.mean(jnp.abs(wz), axis=1) * params.weight
        cost = _apply_power(cost, params.power)
        return cost, {}

    return jax.lax.cond(should_skip, skip_score, do_score, operand=None)


def _apply_power(cost: jax.Array, power: int) -> jax.Array:
    if power > 1:
        return cost**power
    return cost
