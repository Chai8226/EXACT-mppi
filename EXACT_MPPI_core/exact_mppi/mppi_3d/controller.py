from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from .optimizer import Optimizer3D


class MPPIController3D:
    """Public controller facade for yaw-only 3D Core Python MPPI."""

    def __init__(self, *args, **kwargs):
        self.optimizer_ = Optimizer3D(*args, **kwargs)
        self.max_obs_num_ = int(kwargs.get("max_obs_num", 100))
        self.optimal_trajectory_ = None

    def reset(self):
        self.optimizer_.reset()
        self.optimal_trajectory_ = None

    def computeVelocityCommands(
        self,
        robot_pose: ArrayLike,
        robot_speed: ArrayLike,
        plan: ArrayLike,
        goal: ArrayLike,
        obstacle_points: Optional[np.ndarray],
    ) -> np.ndarray:
        packed_points, obstacle_points_mask = self._pack_obstacle_points(
            obstacle_points,
            self.max_obs_num_,
        )

        command, self.optimal_trajectory_ = self.optimizer_.evalControl(
            robot_pose=self._to_jnp(robot_pose),
            robot_speed=self._to_jnp(robot_speed),
            plan=self._to_jnp(plan),
            goal=self._to_jnp(goal),
            obstacle_points=self._to_jnp(packed_points),
            obstacle_points_mask=self._to_jnp(obstacle_points_mask),
        )

        return jax.device_get(command)

    def getOptimalTrajectory(self) -> Optional[np.ndarray]:
        if self.optimal_trajectory_ is None:
            return None
        return jax.device_get(self.optimal_trajectory_)

    def getGeneratedTrajectories(self) -> Optional[np.ndarray]:
        trajectories = self.optimizer_.getGeneratedTrajectories()
        if trajectories is None:
            return None
        return jax.device_get(trajectories)

    def getCosts(self) -> Optional[np.ndarray]:
        costs = self.optimizer_.getCosts()
        if costs is None:
            return None
        return jax.device_get(costs)

    @staticmethod
    def _pack_obstacle_points(
        obstacle_points: Optional[np.ndarray],
        max_points: int = 100,
        dtype=jnp.float32,
    ) -> Tuple[jax.Array, jax.Array]:
        if obstacle_points is None:
            return (
                jnp.zeros((max_points, 3), dtype=dtype),
                jnp.zeros((max_points,), dtype=dtype),
            )

        points = np.asarray(obstacle_points, dtype=np.float32)
        if points.size == 0:
            return (
                jnp.zeros((max_points, 3), dtype=dtype),
                jnp.zeros((max_points,), dtype=dtype),
            )

        points = points.reshape((-1, 3))
        point_count = points.shape[0]

        if point_count > max_points:
            distances = np.einsum("ij,ij->i", points, points)
            nearest_indices = np.argpartition(distances, max_points - 1)[:max_points]
            points = points[nearest_indices]
            point_count = max_points

        packed_points = np.zeros((max_points, 3), dtype=np.float32)
        packed_points[:point_count] = points

        mask = np.zeros((max_points,), dtype=np.float32)
        mask[:point_count] = 1.0

        return jnp.asarray(packed_points, dtype=dtype), jnp.asarray(mask, dtype=dtype)

    @staticmethod
    def _to_jnp(x, dtype=jnp.float32) -> jax.Array:
        return jnp.asarray(x, dtype=dtype)
