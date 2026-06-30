from dataclasses import replace
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from .models import (
    ControlConstraints3D,
    ControlSequence3D,
    OptimizerSettings3D,
    SamplingStd3D,
    Trajectories3D,
    reset_ControlSequence3D,
)
from .motion_models import YawOnly3DHolonomicMotionModel


class Optimizer3D:
    """Minimal MPPI optimizer for the yaw-only 3D Core API."""

    def __init__(
        self,
        model_dt: float = 0.05,
        time_steps: int = 56,
        batch_size: int = 1000,
        iteration_count: int = 1,
        temperature: float = 0.3,
        vx_max: float = 0.5,
        vx_min: float = -0.35,
        vy_max: float = 0.5,
        vz_max: float = 0.5,
        wz_max: float = 1.9,
        vx_std: float = 0.2,
        vy_std: float = 0.2,
        vz_std: float = 0.2,
        wz_std: float = 0.4,
        seed: int = 0,
        debug: bool = False,
        shift_control_sequence: bool = True,
        goal_weight: float = 8.0,
        goal_yaw_weight: float = 0.2,
        path_weight: float = 1.0,
        control_weight: float = 0.02,
        **kwargs,
    ):
        del kwargs
        self.debug_ = debug
        self.settings_ = OptimizerSettings3D(
            constraints=ControlConstraints3D(
                vx_max=abs(vx_max),
                vx_min=float(vx_min),
                vy=abs(vy_max),
                vz=abs(vz_max),
                wz=abs(wz_max),
            ),
            sampling_std=SamplingStd3D(
                vx=float(vx_std),
                vy=float(vy_std),
                vz=float(vz_std),
                wz=float(wz_std),
            ),
            model_dt=float(model_dt),
            temperature=float(temperature),
            batch_size=int(batch_size),
            time_steps=int(time_steps),
            iteration_count=int(iteration_count),
            shift_control_sequence=bool(shift_control_sequence),
            goal_weight=float(goal_weight),
            goal_yaw_weight=float(goal_yaw_weight),
            path_weight=float(path_weight),
            control_weight=float(control_weight),
        )
        self.motion_model_ = YawOnly3DHolonomicMotionModel(model_dt)
        self.reset(seed)

    def reset(self, seed: int = 0):
        self.control_sequence_ = reset_ControlSequence3D(self.settings_.time_steps)
        self.key_ = jax.random.PRNGKey(seed)
        self.last_command_vel_ = jnp.zeros(4, dtype=jnp.float32)
        if self.debug_:
            self.generated_trajectories_ = None
            self.costs_ = None

    def evalControl(
        self,
        robot_pose: jax.Array,
        robot_speed: jax.Array,
        plan: jax.Array,
        goal: jax.Array,
        obstacle_points: jax.Array,
        obstacle_points_mask: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        del robot_speed, obstacle_points, obstacle_points_mask

        self._validate_inputs(robot_pose, plan, goal, self.settings_.time_steps)

        for _ in range(self.settings_.iteration_count):
            (
                self.key_,
                candidate_sequences,
                trajectories,
                costs,
            ) = self._generate_and_score(
                self.key_,
                self.control_sequence_,
                robot_pose,
                plan,
                goal,
            )
            self.control_sequence_ = self._update_control_sequence(
                candidate_sequences,
                costs,
            )

        optimal_trajectory = self.getOptimizedTrajectory(
            self.control_sequence_,
            robot_pose,
        )
        command = self.getControlFromSequence(self.control_sequence_)

        if self.settings_.shift_control_sequence:
            self.control_sequence_ = self.shiftControlSequence(self.control_sequence_)

        self.last_command_vel_ = command

        if self.debug_:
            self.generated_trajectories_ = trajectories
            self.costs_ = costs

        return command, optimal_trajectory

    def _generate_and_score(
        self,
        key: jax.Array,
        control_sequence: ControlSequence3D,
        pose: jax.Array,
        plan: jax.Array,
        goal: jax.Array,
    ) -> Tuple[jax.Array, ControlSequence3D, Trajectories3D, jax.Array]:
        settings = self.settings_
        key, k1, k2, k3, k4 = jax.random.split(key, 5)

        noise_vx = jax.random.normal(k1, (settings.batch_size, settings.time_steps))
        noise_vy = jax.random.normal(k2, (settings.batch_size, settings.time_steps))
        noise_vz = jax.random.normal(k3, (settings.batch_size, settings.time_steps))
        noise_wz = jax.random.normal(k4, (settings.batch_size, settings.time_steps))

        vx = control_sequence.vx[None, :] + noise_vx * settings.sampling_std.vx
        vy = control_sequence.vy[None, :] + noise_vy * settings.sampling_std.vy
        vz = control_sequence.vz[None, :] + noise_vz * settings.sampling_std.vz
        wz = control_sequence.wz[None, :] + noise_wz * settings.sampling_std.wz

        nominal = self._goal_directed_sequence(pose, goal)
        vx = vx.at[0].set(nominal.vx)
        vy = vy.at[0].set(nominal.vy)
        vz = vz.at[0].set(nominal.vz)
        wz = wz.at[0].set(nominal.wz)

        vx, vy, vz, wz = self._clip_controls(vx, vy, vz, wz)
        trajectories = self.motion_model_.integrate_batch(vx, vy, vz, wz, pose)
        costs = self._score_trajectories(trajectories, vx, vy, vz, wz, plan, goal)
        return key, ControlSequence3D(vx=vx, vy=vy, vz=vz, wz=wz), trajectories, costs

    def _goal_directed_sequence(
        self, pose: jax.Array, goal: jax.Array
    ) -> ControlSequence3D:
        horizon = jnp.maximum(
            self.settings_.model_dt * self.settings_.time_steps,
            self.settings_.model_dt,
        )
        delta = goal[:3] - pose[:3]
        yaw = pose[3]
        yaw_cos = jnp.cos(yaw)
        yaw_sin = jnp.sin(yaw)

        vx = (delta[0] * yaw_cos + delta[1] * yaw_sin) / horizon
        vy = (-delta[0] * yaw_sin + delta[1] * yaw_cos) / horizon
        vz = delta[2] / horizon
        wz = self._shortest_angle(goal[3] - pose[3]) / horizon

        vx, vy, vz, wz = self._clip_controls(vx, vy, vz, wz)
        return ControlSequence3D(
            vx=jnp.full((self.settings_.time_steps,), vx),
            vy=jnp.full((self.settings_.time_steps,), vy),
            vz=jnp.full((self.settings_.time_steps,), vz),
            wz=jnp.full((self.settings_.time_steps,), wz),
        )

    def _score_trajectories(
        self,
        trajectories: Trajectories3D,
        vx: jax.Array,
        vy: jax.Array,
        vz: jax.Array,
        wz: jax.Array,
        plan: jax.Array,
        goal: jax.Array,
    ) -> jax.Array:
        final_xyz = jnp.stack(
            [trajectories.x[:, -1], trajectories.y[:, -1], trajectories.z[:, -1]],
            axis=1,
        )
        goal_xyz_cost = jnp.sum((final_xyz - goal[:3]) ** 2, axis=1)
        goal_yaw_cost = self._shortest_angle(trajectories.yaws[:, -1] - goal[3]) ** 2

        path_xyz = plan[: self.settings_.time_steps, :3]
        trajectory_xyz = jnp.stack(
            [trajectories.x, trajectories.y, trajectories.z], axis=2
        )
        path_cost = jnp.mean(
            jnp.sum((trajectory_xyz - path_xyz[None, :, :]) ** 2, axis=2),
            axis=1,
        )

        control_cost = jnp.mean(vx**2 + vy**2 + vz**2 + wz**2, axis=1)

        return (
            self.settings_.goal_weight * goal_xyz_cost
            + self.settings_.goal_yaw_weight * goal_yaw_cost
            + self.settings_.path_weight * path_cost
            + self.settings_.control_weight * control_cost
        )

    def _update_control_sequence(
        self, candidate_sequences: ControlSequence3D, costs: jax.Array
    ) -> ControlSequence3D:
        normalized_costs = costs - jnp.min(costs)
        weights = jax.nn.softmax(-normalized_costs / self.settings_.temperature)
        sequence = ControlSequence3D(
            vx=jnp.sum(candidate_sequences.vx * weights[:, None], axis=0),
            vy=jnp.sum(candidate_sequences.vy * weights[:, None], axis=0),
            vz=jnp.sum(candidate_sequences.vz * weights[:, None], axis=0),
            wz=jnp.sum(candidate_sequences.wz * weights[:, None], axis=0),
        )
        return self.applyControlSequenceConstraints(sequence)

    def applyControlSequenceConstraints(
        self, control_sequence: ControlSequence3D
    ) -> ControlSequence3D:
        vx, vy, vz, wz = self._clip_controls(
            control_sequence.vx,
            control_sequence.vy,
            control_sequence.vz,
            control_sequence.wz,
        )
        return replace(control_sequence, vx=vx, vy=vy, vz=vz, wz=wz)

    def shiftControlSequence(
        self, control_sequence: ControlSequence3D
    ) -> ControlSequence3D:
        return ControlSequence3D(
            vx=jnp.concatenate([control_sequence.vx[1:], control_sequence.vx[-1:]]),
            vy=jnp.concatenate([control_sequence.vy[1:], control_sequence.vy[-1:]]),
            vz=jnp.concatenate([control_sequence.vz[1:], control_sequence.vz[-1:]]),
            wz=jnp.concatenate([control_sequence.wz[1:], control_sequence.wz[-1:]]),
        )

    def getControlFromSequence(self, control_sequence: ControlSequence3D) -> jax.Array:
        offset = 1 if self.settings_.shift_control_sequence else 0
        return jnp.array(
            [
                control_sequence.vx[offset],
                control_sequence.vy[offset],
                control_sequence.vz[offset],
                control_sequence.wz[offset],
            ],
            dtype=jnp.float32,
        )

    def getOptimizedTrajectory(
        self, control_sequence: ControlSequence3D, pose: jax.Array
    ) -> jax.Array:
        return self.motion_model_.integrate_sequence(control_sequence, pose)

    def getGeneratedTrajectories(self) -> Optional[jax.Array]:
        if not hasattr(self, "generated_trajectories_") or self.generated_trajectories_ is None:
            return None
        return jnp.stack(
            (
                self.generated_trajectories_.x,
                self.generated_trajectories_.y,
                self.generated_trajectories_.z,
                self.generated_trajectories_.yaws,
            ),
            axis=-1,
        )

    def getCosts(self) -> Optional[jax.Array]:
        if not hasattr(self, "costs_"):
            return None
        return self.costs_

    def _clip_controls(self, vx, vy, vz, wz):
        constraints = self.settings_.constraints
        return (
            jnp.clip(vx, constraints.vx_min, constraints.vx_max),
            jnp.clip(vy, -constraints.vy, constraints.vy),
            jnp.clip(vz, -constraints.vz, constraints.vz),
            jnp.clip(wz, -constraints.wz, constraints.wz),
        )

    @staticmethod
    def _shortest_angle(angle: jax.Array) -> jax.Array:
        return jnp.arctan2(jnp.sin(angle), jnp.cos(angle))

    @staticmethod
    def _validate_inputs(
        robot_pose: jax.Array,
        plan: jax.Array,
        goal: jax.Array,
        time_steps: int,
    ):
        if robot_pose.shape != (4,):
            raise ValueError("robot_pose must have shape (4,) for [x, y, z, yaw].")
        if goal.shape != (4,):
            raise ValueError("goal must have shape (4,) for [x, y, z, yaw].")
        if plan.ndim != 2 or plan.shape[1] != 4:
            raise ValueError("plan must have shape (T, 4) for [x, y, z, yaw].")
        if plan.shape[0] < time_steps:
            raise ValueError("plan must contain at least time_steps path points.")
