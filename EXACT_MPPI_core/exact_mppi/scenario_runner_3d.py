"""Config-driven yaw-only 3D MPPI scenario runner."""

from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
if "JAX_PLATFORMS" not in os.environ and not Path("/dev/nvidiactl").exists():
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from exact_mppi.mppi_3d import BoxUnionVolume3D, MPPIController3D


@dataclass(frozen=True)
class ScenarioRunResult3D:
    scenario: str
    reached_goal: bool
    collided: bool
    final_distance: float
    minimum_clearance: float | None
    step_count: int
    final_state: np.ndarray
    state_history: np.ndarray
    command_history: np.ndarray
    global_reference_path: np.ndarray
    global_obstacle_points: np.ndarray
    robot_volume_config: list[dict[str, Any]]

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "reached_goal": self.reached_goal,
            "collided": self.collided,
            "final_distance": self.final_distance,
            "minimum_clearance": self.minimum_clearance,
            "step_count": self.step_count,
        }


def load_builtin_scenario_config(name: str) -> dict[str, Any]:
    resource = resources.files("exact_mppi.scenarios_3d").joinpath(f"{name}.yaml")
    if not resource.is_file():
        raise ValueError(f"Unknown built-in 3D scenario: {name}")
    return _load_yaml_text(resource.read_text(encoding="utf-8"))


def load_scenario_config(path: str | Path) -> dict[str, Any]:
    return _load_yaml_text(Path(path).read_text(encoding="utf-8"))


def run_3d_scenario(config: Mapping[str, Any]) -> ScenarioRunResult3D:
    cfg = copy.deepcopy(dict(config))
    scenario_name = str(cfg["name"])
    simulation = dict(cfg.get("simulation", {}))
    model_dt = float(simulation.get("model_dt", 0.15))
    max_steps = int(simulation.get("max_steps", 80))
    goal_tolerance = float(simulation.get("goal_tolerance", 0.28))
    clearance_margin = float(simulation.get("clearance_margin", 0.04))
    observation_range = float(simulation.get("observation_range", 1.7))
    max_obstacle_points = int(simulation.get("max_obstacle_points", 48))

    robot_volume_config = _load_robot_volume_config(cfg)
    robot_volume = BoxUnionVolume3D.from_config(robot_volume_config)
    reference_path = _build_reference_path(cfg["reference_path"])
    obstacle_points = _build_obstacle_points(cfg.get("obstacles", {}))
    controller = _build_controller(
        cfg.get("controller", {}),
        model_dt=model_dt,
        max_obstacle_points=max_obstacle_points,
        robot_volume_config=robot_volume_config,
        clearance_margin=clearance_margin,
    )
    time_steps = int(cfg.get("controller", {}).get("time_steps", 12))

    goal = reference_path[-1].copy()
    state = reference_path[0].copy()
    speed = np.zeros(4, dtype=np.float32)
    state_history = [state.copy()]
    command_history = []
    minimum_clearance = _minimum_state_clearance(
        robot_volume,
        obstacle_points,
        state,
    )

    for _ in range(max_steps):
        local_plan = select_local_plan(reference_path, state, time_steps)
        local_goal = transfer_from_global_to_local_frame(goal[None, :], state)[0]
        local_obstacles = build_range_based_local_observation(
            obstacle_points,
            state,
            observation_range=observation_range,
            max_points=max_obstacle_points,
        )

        command = controller.computeVelocityCommands(
            robot_pose=np.zeros(4, dtype=np.float32),
            robot_speed=speed,
            plan=local_plan,
            goal=local_goal,
            obstacle_points=local_obstacles,
        )
        command = np.asarray(command, dtype=np.float32)

        state = integrate_yaw_only_3d_state(state, command, model_dt)
        speed = command
        state_history.append(state.copy())
        command_history.append(command.copy())

        clearance = _minimum_state_clearance(robot_volume, obstacle_points, state)
        minimum_clearance = _merge_minimum_clearance(minimum_clearance, clearance)

        if np.linalg.norm(state[:3] - goal[:3]) <= goal_tolerance:
            break

    final_distance = float(np.linalg.norm(state[:3] - goal[:3]))
    reached_goal = final_distance <= goal_tolerance
    collided = (
        minimum_clearance is not None and minimum_clearance < clearance_margin
    )

    return ScenarioRunResult3D(
        scenario=scenario_name,
        reached_goal=bool(reached_goal),
        collided=bool(collided),
        final_distance=final_distance,
        minimum_clearance=minimum_clearance,
        step_count=len(command_history),
        final_state=state.copy(),
        state_history=np.asarray(state_history, dtype=np.float32),
        command_history=np.asarray(command_history, dtype=np.float32).reshape((-1, 4)),
        global_reference_path=reference_path,
        global_obstacle_points=obstacle_points,
        robot_volume_config=robot_volume_config,
    )


def build_range_based_local_observation(
    global_obstacle_points: np.ndarray,
    robot_pose: np.ndarray,
    observation_range: float,
    max_points: int,
) -> np.ndarray:
    if global_obstacle_points.size == 0 or max_points <= 0:
        return np.empty((0, 3), dtype=np.float32)

    deltas = global_obstacle_points - robot_pose[:3]
    distances = np.linalg.norm(deltas, axis=1)
    selected_indices = np.flatnonzero(distances <= observation_range)
    if selected_indices.size > max_points:
        nearest_order = np.argsort(distances[selected_indices])[:max_points]
        selected_indices = selected_indices[nearest_order]
    if selected_indices.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    return transfer_from_global_to_local_frame(
        global_obstacle_points[selected_indices],
        robot_pose,
    ).astype(np.float32)


def select_local_plan(
    global_reference_path: np.ndarray,
    robot_pose: np.ndarray,
    time_steps: int,
) -> np.ndarray:
    distances = np.linalg.norm(global_reference_path[:, :3] - robot_pose[:3], axis=1)
    nearest_idx = int(np.argmin(distances))
    end_idx = nearest_idx + time_steps
    if end_idx <= global_reference_path.shape[0]:
        global_plan = global_reference_path[nearest_idx:end_idx]
    else:
        pad_count = end_idx - global_reference_path.shape[0]
        padding = np.repeat(global_reference_path[-1][None, :], pad_count, axis=0)
        global_plan = np.vstack([global_reference_path[nearest_idx:], padding])
    return transfer_from_global_to_local_frame(global_plan, robot_pose)


def transfer_from_global_to_local_frame(
    points: np.ndarray,
    pose: np.ndarray,
) -> np.ndarray:
    p = np.asarray(points, dtype=np.float32)
    pose = np.asarray(pose, dtype=np.float32).reshape(-1)
    out = p.copy()

    c = np.cos(pose[3])
    s = np.sin(pose[3])
    rotation_global_to_local = np.array([[c, -s], [s, c]], dtype=np.float32)

    out[..., :2] = (out[..., :2] - pose[:2]) @ rotation_global_to_local
    if out.shape[-1] >= 3:
        out[..., 2] = out[..., 2] - pose[2]
    if out.shape[-1] >= 4:
        out[..., 3] = _wrap_to_pi(out[..., 3] - pose[3])
    return out


def integrate_yaw_only_3d_state(
    state: np.ndarray,
    command: np.ndarray,
    model_dt: float,
) -> np.ndarray:
    vx, vy, vz, wz = command
    yaw = state[3]
    next_state = state.copy()
    next_state[0] += (vx * np.cos(yaw) - vy * np.sin(yaw)) * model_dt
    next_state[1] += (vx * np.sin(yaw) + vy * np.cos(yaw)) * model_dt
    next_state[2] += vz * model_dt
    next_state[3] = _wrap_to_pi(next_state[3] + wz * model_dt)
    return next_state.astype(np.float32)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a config-driven yaw-only 3D MPPI scenario."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--scenario", default="open_track_3d")
    source.add_argument("--config", type=Path)
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Write the machine-readable summary to this path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = (
        load_scenario_config(args.config)
        if args.config is not None
        else load_builtin_scenario_config(args.scenario)
    )
    result = run_3d_scenario(config)
    summary_json = json.dumps(result.summary, allow_nan=False, sort_keys=True)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(summary_json + "\n", encoding="utf-8")
    else:
        print(summary_json)
    return 0 if result.reached_goal and not result.collided else 1


def _load_yaml_text(text: str) -> dict[str, Any]:
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("3D scenario config must be a YAML mapping.")
    return data


def _load_robot_volume_config(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    robot_volume = config.get("robot_volume", {})
    boxes = (
        robot_volume.get("boxes", robot_volume)
        if isinstance(robot_volume, dict)
        else robot_volume
    )
    if not isinstance(boxes, list) or not boxes:
        raise ValueError("3D scenario config requires at least one robot volume box.")
    return [dict(box) for box in boxes]


def _build_controller(
    controller_config: Mapping[str, Any],
    *,
    model_dt: float,
    max_obstacle_points: int,
    robot_volume_config: list[dict[str, Any]],
    clearance_margin: float,
) -> MPPIController3D:
    kwargs = copy.deepcopy(dict(controller_config))
    kwargs.setdefault("model_dt", model_dt)
    kwargs.setdefault("time_steps", 12)
    kwargs.setdefault("max_obs_num", max_obstacle_points)
    kwargs.setdefault("robot_volume_config", robot_volume_config)
    kwargs.setdefault("obstacles_collision_margin_distance", clearance_margin)
    kwargs.setdefault(
        "TrajectoryValidator",
        {
            "collision_lookahead_time": 1.0,
            "collision_margin_distance": clearance_margin,
        },
    )
    return MPPIController3D(**kwargs)


def _build_reference_path(config: Mapping[str, Any]) -> np.ndarray:
    waypoints = np.asarray(config["waypoints"], dtype=np.float32)
    if waypoints.ndim != 2 or waypoints.shape[1] != 4 or waypoints.shape[0] < 2:
        raise ValueError("3D reference path waypoints must have shape (N, 4).")
    point_count = int(config.get("point_count", waypoints.shape[0]))
    if point_count < 2:
        raise ValueError("3D reference path point_count must be at least 2.")

    segment_lengths = np.linalg.norm(np.diff(waypoints[:, :3], axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    samples = np.linspace(0.0, cumulative[-1], point_count, dtype=np.float32)

    path = np.empty((point_count, 4), dtype=np.float32)
    for dim in range(4):
        path[:, dim] = np.interp(samples, cumulative, waypoints[:, dim])
    return path


def _build_obstacle_points(config: Mapping[str, Any]) -> np.ndarray:
    points = np.asarray(config.get("points", []), dtype=np.float32)
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    return points.reshape((-1, 3)).astype(np.float32)


def _minimum_state_clearance(
    robot_volume: BoxUnionVolume3D,
    global_obstacle_points: np.ndarray,
    robot_pose: np.ndarray,
) -> float | None:
    if global_obstacle_points.size == 0:
        return None
    body_points = transfer_from_global_to_local_frame(
        global_obstacle_points,
        robot_pose,
    )
    distances = robot_volume.signed_distance(
        jnp.asarray(body_points, dtype=jnp.float32)
    )
    return float(jax.device_get(jnp.min(distances)))


def _merge_minimum_clearance(
    current: float | None,
    next_clearance: float | None,
) -> float | None:
    if current is None:
        return next_clearance
    if next_clearance is None:
        return current
    return min(current, next_clearance)


def _wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


if __name__ == "__main__":
    raise SystemExit(main())
