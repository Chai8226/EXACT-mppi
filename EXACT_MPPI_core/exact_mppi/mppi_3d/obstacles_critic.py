from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import tree_util

from .geometry import BoxUnionVolume3D
from .models import Trajectories3D


@tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class ObstaclesCriticParams3D:
    """Obstacle scoring parameters for yaw-only 3D trajectories."""

    enabled: bool
    power: int
    repulsion_weight: float
    critical_weight: float
    collision_cost: float
    collision_margin_distance: float
    repulsion_distance: float


def minimum_signed_distance_from_trajectory_to_obstacle_points(
    trajectory: jax.Array,
    obstacle_points: jax.Array,
    obstacle_points_mask: jax.Array,
    robot_volume: BoxUnionVolume3D,
    masked_clearance: float = 1.0e6,
) -> jax.Array:
    """Return minimum SDF clearance for each trajectory pose.

    The trajectory and obstacle points are expected to be expressed in the same
    local yaw-only 3D frame. Each pose transforms obstacle points into the robot
    body frame before evaluating the configured box-union volume SDF.
    """

    points = jnp.asarray(obstacle_points, dtype=jnp.float32)
    mask = jnp.asarray(obstacle_points_mask, dtype=jnp.float32) > 0.0

    def distance_at_pose(pose):
        xyz = pose[:3]
        yaw = pose[3]
        rel = points - xyz[None, :]
        yaw_cos = jnp.cos(yaw)
        yaw_sin = jnp.sin(yaw)
        body_points = jnp.stack(
            [
                rel[:, 0] * yaw_cos + rel[:, 1] * yaw_sin,
                -rel[:, 0] * yaw_sin + rel[:, 1] * yaw_cos,
                rel[:, 2],
            ],
            axis=1,
        )
        distances = robot_volume.signed_distance(body_points)
        masked_distances = jnp.where(mask, distances, masked_clearance)
        return jnp.min(masked_distances)

    return jax.vmap(distance_at_pose)(trajectory)


def minimum_signed_distance_from_trajectories_to_obstacle_points(
    trajectories: Trajectories3D,
    obstacle_points: jax.Array,
    obstacle_points_mask: jax.Array,
    robot_volume: BoxUnionVolume3D,
) -> jax.Array:
    trajectory_array = jnp.stack(
        [trajectories.x, trajectories.y, trajectories.z, trajectories.yaws],
        axis=2,
    )
    return jax.vmap(
        minimum_signed_distance_from_trajectory_to_obstacle_points,
        in_axes=(0, None, None, None),
    )(trajectory_array, obstacle_points, obstacle_points_mask, robot_volume)


def obstacles_critic_score_3d(
    trajectories: Trajectories3D,
    obstacle_points: jax.Array,
    obstacle_points_mask: jax.Array,
    robot_volume: BoxUnionVolume3D,
    params: ObstaclesCriticParams3D,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Score trajectories by exact 3D robot-volume clearance."""

    if not params.enabled:
        batch_size = trajectories.x.shape[0]
        time_steps = trajectories.x.shape[1]
        return (
            jnp.zeros((batch_size,), dtype=jnp.float32),
            jnp.full((batch_size, time_steps), 1.0e6, dtype=jnp.float32),
            jnp.array(False),
        )

    dist_to_obj = minimum_signed_distance_from_trajectories_to_obstacle_points(
        trajectories,
        obstacle_points,
        obstacle_points_mask,
        robot_volume,
    )

    trajectory_collide = jnp.any(dist_to_obj < 0.0, axis=1)
    critical_cost = jnp.sum(
        jax.nn.relu(params.collision_margin_distance - dist_to_obj),
        axis=1,
    )
    repulsive_cost = jnp.mean(
        jax.nn.relu(params.repulsion_distance - dist_to_obj),
        axis=1,
    )

    raw_cost = jnp.where(trajectory_collide, params.collision_cost, critical_cost)
    cost = params.critical_weight * raw_cost + params.repulsion_weight * repulsive_cost

    if params.power > 1:
        cost = cost**params.power

    fail_flag = jnp.all(trajectory_collide)
    return cost, dist_to_obj, fail_flag
