from enum import IntEnum
from typing import Tuple

import jax
import jax.numpy as jnp

from .geometry import BoxUnionVolume3D
from .models import OptimizerSettings3D
from .obstacles_critic import (
    minimum_signed_distance_from_trajectory_to_obstacle_points,
)


class ValidationResult3D(IntEnum):
    SUCCESS = 0
    SOFT_RESET = 1
    FAILURE = 2


class OptimalTrajectoryValidator3D:
    """Validate yaw-only 3D optimal trajectories against robot-volume clearance."""

    def __init__(
        self,
        settings: OptimizerSettings3D,
        collision_lookahead_time: float = 2.0,
        collision_margin_distance: float = 0.1,
    ):
        self.collision_lookahead_time_ = float(collision_lookahead_time)
        self.traj_samples_to_evaluate_ = int(
            self.collision_lookahead_time_ / settings.model_dt
        )
        self.traj_samples_to_evaluate_ = min(
            self.traj_samples_to_evaluate_,
            int(settings.time_steps),
        )
        self.traj_samples_to_evaluate_ = max(self.traj_samples_to_evaluate_, 1)
        self.collision_margin_distance_ = max(0.0, float(collision_margin_distance))

    def validateTrajectory(
        self,
        optimal_trajectory: jax.Array,
        obstacle_points: jax.Array,
        obstacle_points_mask: jax.Array,
        robot_volume: BoxUnionVolume3D,
    ) -> Tuple[jax.Array, jax.Array]:
        subset = optimal_trajectory[: self.traj_samples_to_evaluate_]
        dist_t = minimum_signed_distance_from_trajectory_to_obstacle_points(
            subset,
            obstacle_points,
            obstacle_points_mask,
            robot_volume,
        )
        dist_min = jnp.min(dist_t)
        unsafe = jnp.any(dist_t < self.collision_margin_distance_)
        result = jnp.where(
            unsafe,
            jnp.int32(ValidationResult3D.SOFT_RESET),
            jnp.int32(ValidationResult3D.SUCCESS),
        )
        return result, dist_min
