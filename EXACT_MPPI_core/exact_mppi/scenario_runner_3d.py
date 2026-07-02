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


STATIC_3D_SCENARIOS = (
    "open_track_3d",
    "narrow_gap_t_volume_3d",
    "vertical_gate_3d",
    "t_shape_trap_3d",
    "cluttered_corridor_3d",
)


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
    local_plan_history: np.ndarray
    optimal_trajectory_history: np.ndarray
    clearance_history: list[float | None]
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
            **compute_3d_smoothness_telemetry(
                command_history=self.command_history,
                state_history=self.state_history,
            ),
        }


def load_builtin_scenario_config(name: str) -> dict[str, Any]:
    resource = resources.files("exact_mppi.scenarios_3d").joinpath(f"{name}.yaml")
    if not resource.is_file():
        raise ValueError(f"Unknown built-in 3D scenario: {name}")
    return _load_yaml_text(resource.read_text(encoding="utf-8"))


def load_scenario_config(path: str | Path) -> dict[str, Any]:
    return _load_yaml_text(Path(path).read_text(encoding="utf-8"))


def run_3d_static_scenario_suite(
    scenario_names: list[str] | tuple[str, ...] | None = None,
) -> list[ScenarioRunResult3D]:
    names = STATIC_3D_SCENARIOS if scenario_names is None else tuple(scenario_names)
    return [run_3d_scenario(load_builtin_scenario_config(name)) for name in names]


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
    local_plan_history = []
    optimal_trajectory_history = []
    minimum_clearance = _minimum_state_clearance(
        robot_volume,
        obstacle_points,
        state,
    )
    clearance_history = [minimum_clearance]

    for _ in range(max_steps):
        global_plan = select_global_plan(reference_path, state, time_steps)
        local_plan = transfer_from_global_to_local_frame(global_plan, state)
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
        optimal_trajectory = controller.getOptimalTrajectory()
        if optimal_trajectory is None:
            global_optimal_trajectory = np.empty((0, 4), dtype=np.float32)
        else:
            global_optimal_trajectory = transfer_from_local_to_global_frame(
                np.asarray(optimal_trajectory, dtype=np.float32),
                state,
            )

        state = integrate_yaw_only_3d_state(state, command, model_dt)
        speed = command
        state_history.append(state.copy())
        command_history.append(command.copy())
        local_plan_history.append(global_plan.copy())
        optimal_trajectory_history.append(global_optimal_trajectory.copy())

        clearance = _minimum_state_clearance(robot_volume, obstacle_points, state)
        minimum_clearance = _merge_minimum_clearance(minimum_clearance, clearance)
        clearance_history.append(clearance)

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
        local_plan_history=np.asarray(local_plan_history, dtype=np.float32),
        optimal_trajectory_history=np.asarray(
            optimal_trajectory_history,
            dtype=np.float32,
        ),
        clearance_history=clearance_history,
        global_reference_path=reference_path,
        global_obstacle_points=obstacle_points,
        robot_volume_config=robot_volume_config,
    )


def build_3d_replay_data(result: ScenarioRunResult3D) -> dict[str, Any]:
    goal = result.global_reference_path[-1]
    frames = []
    for idx, command in enumerate(result.command_history):
        state = result.state_history[idx + 1]
        frame_command_history = result.command_history[: idx + 1]
        frame_state_history = result.state_history[: idx + 2]
        frames.append(
            {
                "frame_index": idx,
                "state": state.tolist(),
                "executed_path": frame_state_history.tolist(),
                "local_plan": result.local_plan_history[idx].tolist(),
                "optimal_trajectory": result.optimal_trajectory_history[idx].tolist(),
                "command": command.tolist(),
                "clearance": result.clearance_history[idx + 1],
                "goal_distance": float(np.linalg.norm(state[:3] - goal[:3])),
                "smoothness_telemetry": compute_3d_smoothness_telemetry(
                    command_history=frame_command_history,
                    state_history=frame_state_history,
                ),
            }
        )

    return {
        "schema_version": 1,
        "summary": result.summary,
        "scene": {
            "scenario": result.scenario,
            "coordinate_conventions": {
                "frame": "world",
                "state": "[x, y, z, yaw]",
                "command": "[vx, vy, vz, wz]",
                "yaw_unit": "radians",
            },
            "reference_path": result.global_reference_path.tolist(),
            "obstacle_points": result.global_obstacle_points.tolist(),
            "robot_volume": {
                "type": "box_union",
                "boxes": result.robot_volume_config,
            },
        },
        "frames": frames,
    }


def write_3d_replay_json(
    result: ScenarioRunResult3D,
    path: str | Path,
) -> None:
    replay_json = json.dumps(
        build_3d_replay_data(result),
        allow_nan=False,
        sort_keys=True,
    )
    replay_path = Path(path)
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    replay_path.write_text(replay_json + "\n", encoding="utf-8")


def compute_3d_smoothness_telemetry(
    *,
    command_history: np.ndarray,
    state_history: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    """Compute step-history smoothness metrics for yaw-only 3D scenario runs.

    The metrics are derived only from executed scenario histories. They do not
    use viewer frame rate, playback interpolation, or wall-clock timing.

    Command smoothness is measured as L2 norms of adjacent command deltas in
    `[vx, vy, vz, wz]`. Trajectory smoothness is measured as L2 norms of second
    differences in executed `[x, y, z, yaw]` states after yaw unwrapping.
    """

    commands = _as_history_matrix(command_history, width=4, name="command_history")
    states = _as_history_matrix(state_history, width=4, name="state_history")

    command_deltas = np.diff(commands, axis=0)
    command_delta_norms = np.linalg.norm(command_deltas, axis=1)

    unwrapped_states = states.copy()
    if unwrapped_states.shape[0] > 0:
        unwrapped_states[:, 3] = np.unwrap(unwrapped_states[:, 3])
    trajectory_second_differences = np.diff(unwrapped_states, n=2, axis=0)
    trajectory_second_difference_norms = np.linalg.norm(
        trajectory_second_differences,
        axis=1,
    )

    return {
        "command_smoothness": _norm_metrics(
            command_delta_norms,
            sample_count_name="sample_count",
            mean_name="mean_delta_norm",
            rms_name="rms_delta_norm",
            max_name="max_delta_norm",
            total_name="total_delta_norm",
        ),
        "trajectory_smoothness": _norm_metrics(
            trajectory_second_difference_norms,
            sample_count_name="sample_count",
            mean_name="mean_second_difference_norm",
            rms_name="rms_second_difference_norm",
            max_name="max_second_difference_norm",
            total_name="total_second_difference_norm",
        ),
    }


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
    return transfer_from_global_to_local_frame(
        select_global_plan(global_reference_path, robot_pose, time_steps),
        robot_pose,
    )


def select_global_plan(
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
    return global_plan.astype(np.float32)


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


def transfer_from_local_to_global_frame(
    points: np.ndarray,
    pose: np.ndarray,
) -> np.ndarray:
    p = np.asarray(points, dtype=np.float32)
    pose = np.asarray(pose, dtype=np.float32).reshape(-1)
    out = p.copy()

    c = np.cos(pose[3])
    s = np.sin(pose[3])
    rotation_local_to_global = np.array([[c, s], [-s, c]], dtype=np.float32)

    out[..., :2] = out[..., :2] @ rotation_local_to_global + pose[:2]
    if out.shape[-1] >= 3:
        out[..., 2] = out[..., 2] + pose[2]
    if out.shape[-1] >= 4:
        out[..., 3] = _wrap_to_pi(out[..., 3] + pose[3])
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
    source.add_argument(
        "--all-static",
        action="store_true",
        help="Run every built-in static 3D scenario and emit a summary list.",
    )
    source.add_argument("--config", type=Path)
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Write the machine-readable summary to this path.",
    )
    parser.add_argument(
        "--replay-json",
        type=Path,
        help="Write Offline Web replay JSON for the selected scenario.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.all_static:
        results = run_3d_static_scenario_suite()
        summaries = [result.summary for result in results]
        summary_json = json.dumps(summaries, allow_nan=False, sort_keys=True)
        exit_ok = True
    else:
        config = (
            load_scenario_config(args.config)
            if args.config is not None
            else load_builtin_scenario_config(args.scenario)
        )
        result = run_3d_scenario(config)
        summary_json = json.dumps(result.summary, allow_nan=False, sort_keys=True)
        exit_ok = result.reached_goal and not result.collided
        if args.replay_json is not None:
            write_3d_replay_json(result, args.replay_json)

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(summary_json + "\n", encoding="utf-8")
    else:
        print(summary_json)
    return 0 if exit_ok else 1


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


def _as_history_matrix(
    history: np.ndarray,
    *,
    width: int,
    name: str,
) -> np.ndarray:
    arr = np.asarray(history, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, width), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != width:
        raise ValueError(f"{name} must have shape (N, {width}).")
    return arr


def _norm_metrics(
    values: np.ndarray,
    *,
    sample_count_name: str,
    mean_name: str,
    rms_name: str,
    max_name: str,
    total_name: str,
) -> dict[str, float | int]:
    if values.size == 0:
        return {
            sample_count_name: 0,
            mean_name: 0.0,
            rms_name: 0.0,
            max_name: 0.0,
            total_name: 0.0,
        }

    return {
        sample_count_name: int(values.size),
        mean_name: float(np.mean(values)),
        rms_name: float(np.sqrt(np.mean(values**2))),
        max_name: float(np.max(values)),
        total_name: float(np.sum(values)),
    }


def _wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


if __name__ == "__main__":
    raise SystemExit(main())
