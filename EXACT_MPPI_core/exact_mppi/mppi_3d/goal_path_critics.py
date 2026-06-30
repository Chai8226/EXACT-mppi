from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import tree_util

from .models import ControlConstraints3D, Trajectories3D


@tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class ConstraintCriticParams3D:
    """Constraint scoring parameters for yaw-only 3D controls."""

    enabled: bool
    power: int
    weight: float


@tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class GoalCriticParams3D:
    """Goal-position scoring parameters for yaw-only 3D trajectories."""

    enabled: bool
    power: int
    weight: float


@tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class GoalYawCriticParams3D:
    """Near-goal yaw scoring parameters for yaw-only 3D trajectories."""

    enabled: bool
    power: int
    weight: float
    threshold_to_consider: float


@tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class PathAlignCriticParams3D:
    """Position-only path-alignment scoring parameters for 3D trajectories."""

    enabled: bool
    power: int
    weight: float
    trajectory_point_step: int


@tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class PathFollowCriticParams3D:
    """3D path-progress scoring parameters for yaw-only 3D trajectories."""

    enabled: bool
    power: int
    weight: float
    offset_from_nearest: int


def constraint_critic_initialize_3d(
    critic_params_dict: dict,
) -> ConstraintCriticParams3D:
    return ConstraintCriticParams3D(
        enabled=critic_params_dict.get("enabled", True),
        power=critic_params_dict.get("cost_power", 1),
        weight=critic_params_dict.get("cost_weight", 4.0),
    )


def goal_critic_initialize_3d(critic_params_dict: dict) -> GoalCriticParams3D:
    return GoalCriticParams3D(
        enabled=critic_params_dict.get("enabled", True),
        power=critic_params_dict.get("cost_power", 1),
        weight=critic_params_dict.get("cost_weight", 5.0),
    )


def goal_yaw_critic_initialize_3d(critic_params_dict: dict) -> GoalYawCriticParams3D:
    return GoalYawCriticParams3D(
        enabled=critic_params_dict.get("enabled", True),
        power=critic_params_dict.get("cost_power", 1),
        weight=critic_params_dict.get("cost_weight", 3.0),
        threshold_to_consider=critic_params_dict.get("threshold_to_consider", 0.5),
    )


def path_align_critic_initialize_3d(
    critic_params_dict: dict,
) -> PathAlignCriticParams3D:
    return PathAlignCriticParams3D(
        enabled=critic_params_dict.get("enabled", True),
        power=critic_params_dict.get("cost_power", 1),
        weight=critic_params_dict.get("cost_weight", 10.0),
        trajectory_point_step=critic_params_dict.get("trajectory_point_step", 4),
    )


def path_follow_critic_initialize_3d(
    critic_params_dict: dict,
) -> PathFollowCriticParams3D:
    return PathFollowCriticParams3D(
        enabled=critic_params_dict.get("enabled", True),
        power=critic_params_dict.get("cost_power", 1),
        weight=critic_params_dict.get("cost_weight", 5.0),
        offset_from_nearest=critic_params_dict.get(
            "offset_from_nearest",
            critic_params_dict.get("offset_from_furthest", 6),
        ),
    )


def constraint_critic_score_3d(
    vx: jax.Array,
    vy: jax.Array,
    vz: jax.Array,
    wz: jax.Array,
    constraints: ControlConstraints3D,
    params: ConstraintCriticParams3D,
    model_dt: float,
) -> Tuple[jax.Array, dict]:
    """Score 4D controls that exceed yaw-only 3D control limits."""

    batch_size = vx.shape[0]
    if not params.enabled:
        return jnp.zeros((batch_size,), dtype=jnp.float32), {}

    violation = (
        jax.nn.relu(vx - constraints.vx_max)
        + jax.nn.relu(constraints.vx_min - vx)
        + jax.nn.relu(jnp.abs(vy) - constraints.vy)
        + jax.nn.relu(jnp.abs(vz) - constraints.vz)
        + jax.nn.relu(jnp.abs(wz) - constraints.wz)
    )
    cost = jnp.sum(violation * model_dt, axis=1) * params.weight
    cost = _apply_power(cost, params.power)
    return cost, {}


def goal_critic_score_3d(
    trajectories: Trajectories3D,
    goal: jax.Array,
    params: GoalCriticParams3D,
) -> Tuple[jax.Array, dict]:
    """Score 3D distance from each trajectory to the goal position."""

    batch_size = trajectories.x.shape[0]
    if not params.enabled:
        return jnp.zeros((batch_size,), dtype=jnp.float32), {}

    trajectory_xyz = _trajectory_xyz(trajectories)
    delta_xyz = trajectory_xyz - goal[:3][None, None, :]
    cost = jnp.mean(jnp.linalg.norm(delta_xyz, axis=2), axis=1) * params.weight
    cost = _apply_power(cost, params.power)
    return cost, {}


def goal_yaw_critic_score_3d(
    trajectories: Trajectories3D,
    current_pose: jax.Array,
    goal: jax.Array,
    params: GoalYawCriticParams3D,
) -> Tuple[jax.Array, dict]:
    """Score yaw only when the current 3D position is near the goal position."""

    batch_size = trajectories.x.shape[0]
    if not params.enabled:
        return jnp.zeros((batch_size,), dtype=jnp.float32), {}

    distance_to_goal = jnp.linalg.norm(current_pose[:3] - goal[:3])
    if distance_to_goal > params.threshold_to_consider:
        return jnp.zeros((batch_size,), dtype=jnp.float32), {}

    dyaw = _shortest_angle(trajectories.yaws - goal[3])
    cost = jnp.mean(jnp.abs(dyaw), axis=1) * params.weight
    cost = _apply_power(cost, params.power)
    return cost, {}


def path_align_critic_score_3d(
    trajectories: Trajectories3D,
    plan: jax.Array,
    params: PathAlignCriticParams3D,
) -> Tuple[jax.Array, dict]:
    """Score position-only distance from sampled trajectory points to the 3D path."""

    batch_size = trajectories.x.shape[0]
    if (not params.enabled) or plan.shape[0] == 0:
        return jnp.zeros((batch_size,), dtype=jnp.float32), {}

    step = max(int(params.trajectory_point_step), 1)
    sampled_xyz = _trajectory_xyz(trajectories)[:, ::step, :]
    path_xyz = plan[:, :3]

    deltas = sampled_xyz[:, :, None, :] - path_xyz[None, None, :, :]
    nearest_distances = jnp.min(jnp.linalg.norm(deltas, axis=3), axis=2)
    cost = jnp.mean(nearest_distances, axis=1) * params.weight
    cost = _apply_power(cost, params.power)
    return cost, {}


def path_follow_critic_score_3d(
    trajectories: Trajectories3D,
    current_pose: jax.Array,
    plan: jax.Array,
    params: PathFollowCriticParams3D,
) -> Tuple[jax.Array, dict]:
    """Score the final trajectory position against a forward point on the 3D path."""

    batch_size = trajectories.x.shape[0]
    if (not params.enabled) or plan.shape[0] == 0:
        return (
            jnp.zeros((batch_size,), dtype=jnp.float32),
            {"path_follow_point": jnp.full((3,), jnp.nan, dtype=jnp.float32)},
        )

    path_xyz = plan[:, :3]
    nearest_idx = jnp.argmin(jnp.linalg.norm(path_xyz - current_pose[:3], axis=1))
    target_idx = jnp.minimum(
        nearest_idx + jnp.asarray(params.offset_from_nearest, dtype=jnp.int32),
        jnp.asarray(plan.shape[0] - 1, dtype=jnp.int32),
    )
    target_xyz = path_xyz[target_idx]

    final_xyz = jnp.stack(
        [trajectories.x[:, -1], trajectories.y[:, -1], trajectories.z[:, -1]],
        axis=1,
    )
    cost = jnp.linalg.norm(final_xyz - target_xyz[None, :], axis=1) * params.weight
    cost = _apply_power(cost, params.power)
    return cost, {"path_follow_point": target_xyz}


def _trajectory_xyz(trajectories: Trajectories3D) -> jax.Array:
    return jnp.stack([trajectories.x, trajectories.y, trajectories.z], axis=2)


def _shortest_angle(angle: jax.Array) -> jax.Array:
    return jnp.arctan2(jnp.sin(angle), jnp.cos(angle))


def _apply_power(cost: jax.Array, power: int) -> jax.Array:
    if power > 1:
        return cost**power
    return cost
