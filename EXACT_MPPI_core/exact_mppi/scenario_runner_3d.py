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
BASELINE_SMOOTHNESS_REFERENCE_SCENARIO = "open_track_3d"
BASELINE_SMOOTHNESS_MULTIPLIER = 2.0
BASELINE_SMOOTHNESS_EPSILON = 1e-6
DEFAULT_REFERENCE_WINDOW_LOOKAHEAD_DISTANCE = 1.2


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
    rollout_history: list[np.ndarray]
    clearance_history: list[float | None]
    global_reference_path: np.ndarray
    global_obstacle_points: np.ndarray
    observed_point_cloud_history: list[np.ndarray]
    obstacle_geometry_config: list[dict[str, Any]]
    robot_volume_config: list[dict[str, Any]]
    sensor_config: dict[str, Any] | None = None

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
    *,
    collect_rollouts: bool = False,
    max_rollouts: int = 8,
) -> list[ScenarioRunResult3D]:
    names = STATIC_3D_SCENARIOS if scenario_names is None else tuple(scenario_names)
    return [
        run_3d_scenario(
            load_builtin_scenario_config(name),
            collect_rollouts=collect_rollouts,
            max_rollouts=max_rollouts,
        )
        for name in names
    ]


def build_3d_baseline_report(
    results: list[ScenarioRunResult3D] | tuple[ScenarioRunResult3D, ...],
    *,
    replay_artifacts: Mapping[str, str] | None = None,
    smoothness_multiplier: float = BASELINE_SMOOTHNESS_MULTIPLIER,
) -> dict[str, Any]:
    scenario_reports = []
    passed_scenarios = []
    failed_scenarios = []
    collided_scenarios = []
    missed_goal_scenarios = []
    poor_smoothness_scenarios = []
    needs_followup_tuning_scenarios = []

    reference = _baseline_smoothness_reference(results)
    command_limit = _baseline_command_smoothness_limit(
        reference,
        smoothness_multiplier=smoothness_multiplier,
    )
    trajectory_limit = _baseline_trajectory_smoothness_limit(
        reference,
        smoothness_multiplier=smoothness_multiplier,
    )

    for result in results:
        summary = copy.deepcopy(result.summary)
        scenario_name = result.scenario
        status = "pass" if result.reached_goal and not result.collided else "fail"
        reasons = []
        if not result.reached_goal:
            reasons.append("missed_goal")
        if result.collided:
            reasons.append("collision")
        if _baseline_command_smoothness_value(result) > command_limit:
            reasons.append("poor_command_smoothness")
        if _baseline_trajectory_smoothness_value(result) > trajectory_limit:
            reasons.append("poor_trajectory_smoothness")

        if status == "pass":
            passed_scenarios.append(scenario_name)
        else:
            failed_scenarios.append(scenario_name)
        if result.collided:
            collided_scenarios.append(scenario_name)
        if not result.reached_goal:
            missed_goal_scenarios.append(scenario_name)
        if (
            "poor_command_smoothness" in reasons
            or "poor_trajectory_smoothness" in reasons
        ):
            poor_smoothness_scenarios.append(scenario_name)
        if reasons:
            needs_followup_tuning_scenarios.append(scenario_name)

        summary["status"] = status
        summary["needs_followup_tuning"] = bool(reasons)
        summary["followup_tuning_reasons"] = reasons
        if replay_artifacts is not None and scenario_name in replay_artifacts:
            summary["replay_json"] = replay_artifacts[scenario_name]
        scenario_reports.append(summary)

    return {
        "schema_version": 1,
        "kind": "3d_static_scenario_baseline",
        "smoothness_reference": {
            "scenario": reference.scenario if reference is not None else None,
            "multiplier": smoothness_multiplier,
            "command_max_delta_norm_limit": command_limit,
            "trajectory_max_second_difference_norm_limit": trajectory_limit,
        },
        "aggregate": {
            "scenario_count": len(scenario_reports),
            "pass_count": len(passed_scenarios),
            "fail_count": len(failed_scenarios),
            "collision_count": len(collided_scenarios),
            "missed_goal_count": len(missed_goal_scenarios),
            "poor_smoothness_count": len(poor_smoothness_scenarios),
            "needs_followup_tuning_count": len(needs_followup_tuning_scenarios),
            "passed_scenarios": passed_scenarios,
            "failed_scenarios": failed_scenarios,
            "collided_scenarios": collided_scenarios,
            "missed_goal_scenarios": missed_goal_scenarios,
            "poor_smoothness_scenarios": poor_smoothness_scenarios,
            "needs_followup_tuning_scenarios": needs_followup_tuning_scenarios,
        },
        "scenarios": scenario_reports,
    }


def write_3d_baseline_replay_artifacts(
    results: list[ScenarioRunResult3D] | tuple[ScenarioRunResult3D, ...],
    directory: str | Path,
    *,
    include_rollouts: bool = False,
    max_rollouts: int = 8,
) -> dict[str, str]:
    replay_dir = Path(directory)
    replay_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for result in results:
        replay_path = replay_dir / f"{result.scenario}.replay.json"
        write_3d_replay_json(
            result,
            replay_path,
            include_rollouts=include_rollouts,
            max_rollouts=max_rollouts,
        )
        artifacts[result.scenario] = str(replay_path)
    return artifacts


def run_3d_scenario(
    config: Mapping[str, Any],
    *,
    collect_rollouts: bool = False,
    max_rollouts: int = 8,
) -> ScenarioRunResult3D:
    cfg = copy.deepcopy(dict(config))
    scenario_name = str(cfg["name"])
    simulation = dict(cfg.get("simulation", {}))
    model_dt = float(simulation.get("model_dt", 0.15))
    max_steps = int(simulation.get("max_steps", 80))
    goal_tolerance = float(simulation.get("goal_tolerance", 0.28))
    clearance_margin = float(simulation.get("clearance_margin", 0.04))
    reference_window_lookahead_distance = float(
        simulation.get(
            "reference_window_lookahead_distance",
            DEFAULT_REFERENCE_WINDOW_LOOKAHEAD_DISTANCE,
        )
    )

    robot_volume_config = _load_robot_volume_config(cfg)
    robot_volume = BoxUnionVolume3D.from_config(robot_volume_config)
    reference_path = _build_reference_path(cfg["reference_path"])
    obstacle_config = cfg.get("obstacles", {})
    obstacle_points = _build_obstacle_points(obstacle_config)
    obstacle_geometry_config = _build_obstacle_geometry_config(obstacle_config)
    sensor_config = _build_sensor_config(cfg.get("sensor"))
    controller_config = dict(cfg.get("controller", {}))
    controller_point_budget = _controller_point_budget(
        controller_config,
    )
    _validate_observation_config(
        sensor_config=sensor_config,
    )
    if collect_rollouts:
        controller_config["debug"] = True
    controller = _build_controller(
        controller_config,
        model_dt=model_dt,
        controller_point_budget=controller_point_budget,
        robot_volume_config=robot_volume_config,
        clearance_margin=clearance_margin,
    )
    time_steps = int(controller_config.get("time_steps", 12))

    goal = reference_path[-1].copy()
    state = reference_path[0].copy()
    speed = np.zeros(4, dtype=np.float32)
    state_history = [state.copy()]
    command_history = []
    local_plan_history = []
    optimal_trajectory_history = []
    rollout_history = []
    observed_point_cloud_history = []
    minimum_clearance = _minimum_state_clearance(
        robot_volume,
        robot_volume_config,
        obstacle_geometry_config,
        obstacle_points,
        state,
    )
    clearance_history = [minimum_clearance]

    for _ in range(max_steps):
        global_plan = select_global_plan(
            reference_path,
            state,
            time_steps,
            reference_window_lookahead_distance=reference_window_lookahead_distance,
        )
        local_plan = transfer_from_global_to_local_frame(global_plan, state)
        local_goal = transfer_from_global_to_local_frame(goal[None, :], state)[0]
        observed_point_cloud = _build_observed_point_cloud_for_step(
            sensor_config=sensor_config,
            obstacle_geometry_config=obstacle_geometry_config,
            robot_pose=state,
        )
        local_obstacles = transfer_from_global_to_local_frame(
            observed_point_cloud,
            state,
        ).astype(np.float32)

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
        global_rollouts = np.empty((0, 0, 4), dtype=np.float32)
        if collect_rollouts:
            generated_rollouts = controller.getGeneratedTrajectories()
            global_rollouts = _sample_global_rollouts(
                generated_rollouts,
                state,
                max_rollouts=max_rollouts,
            )

        state = integrate_yaw_only_3d_state(state, command, model_dt)
        speed = command
        state_history.append(state.copy())
        command_history.append(command.copy())
        local_plan_history.append(global_plan.copy())
        optimal_trajectory_history.append(global_optimal_trajectory.copy())
        observed_point_cloud_history.append(observed_point_cloud.copy())
        if collect_rollouts:
            rollout_history.append(global_rollouts.copy())

        clearance = _minimum_state_clearance(
            robot_volume,
            robot_volume_config,
            obstacle_geometry_config,
            obstacle_points,
            state,
        )
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
        rollout_history=rollout_history,
        clearance_history=clearance_history,
        global_reference_path=reference_path,
        global_obstacle_points=obstacle_points,
        observed_point_cloud_history=observed_point_cloud_history,
        obstacle_geometry_config=obstacle_geometry_config,
        robot_volume_config=robot_volume_config,
        sensor_config=sensor_config,
    )


def build_3d_replay_data(
    result: ScenarioRunResult3D,
    *,
    include_rollouts: bool = False,
    max_rollouts: int = 8,
) -> dict[str, Any]:
    goal = result.global_reference_path[-1]
    frames = []
    for idx, command in enumerate(result.command_history):
        state = result.state_history[idx + 1]
        frame_command_history = result.command_history[: idx + 1]
        frame_state_history = result.state_history[: idx + 2]
        frame = {
            "frame_index": idx,
            "state": state.tolist(),
            "executed_path": frame_state_history.tolist(),
            "reference_window": result.local_plan_history[idx].tolist(),
            "local_plan": result.local_plan_history[idx].tolist(),
            "optimal_trajectory": result.optimal_trajectory_history[idx].tolist(),
            "observed_point_cloud": result.observed_point_cloud_history[idx].tolist(),
            "command": command.tolist(),
            "clearance": result.clearance_history[idx + 1],
            "goal_distance": float(np.linalg.norm(state[:3] - goal[:3])),
            "smoothness_telemetry": compute_3d_smoothness_telemetry(
                command_history=frame_command_history,
                state_history=frame_state_history,
            ),
        }
        if include_rollouts:
            frame["rollouts"] = _bounded_rollouts_for_frame(
                result.rollout_history,
                idx,
                max_rollouts=max_rollouts,
            )
        frames.append(frame)

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
            "obstacle_geometry": result.obstacle_geometry_config,
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
    *,
    include_rollouts: bool = False,
    max_rollouts: int = 8,
) -> None:
    replay_json = json.dumps(
        build_3d_replay_data(
            result,
            include_rollouts=include_rollouts,
            max_rollouts=max_rollouts,
        ),
        allow_nan=False,
        sort_keys=True,
    )
    replay_path = Path(path)
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    replay_path.write_text(replay_json + "\n", encoding="utf-8")


def _baseline_smoothness_reference(
    results: list[ScenarioRunResult3D] | tuple[ScenarioRunResult3D, ...],
) -> ScenarioRunResult3D | None:
    if not results:
        return None
    for result in results:
        if result.scenario == BASELINE_SMOOTHNESS_REFERENCE_SCENARIO:
            return result
    return results[0]


def _baseline_command_smoothness_limit(
    reference: ScenarioRunResult3D | None,
    *,
    smoothness_multiplier: float,
) -> float:
    if reference is None:
        return 0.0
    return max(
        _baseline_command_smoothness_value(reference) * smoothness_multiplier,
        BASELINE_SMOOTHNESS_EPSILON,
    )


def _baseline_trajectory_smoothness_limit(
    reference: ScenarioRunResult3D | None,
    *,
    smoothness_multiplier: float,
) -> float:
    if reference is None:
        return 0.0
    return max(
        _baseline_trajectory_smoothness_value(reference) * smoothness_multiplier,
        BASELINE_SMOOTHNESS_EPSILON,
    )


def _baseline_command_smoothness_value(result: ScenarioRunResult3D) -> float:
    return float(result.summary["command_smoothness"]["max_delta_norm"])


def _baseline_trajectory_smoothness_value(result: ScenarioRunResult3D) -> float:
    return float(
        result.summary["trajectory_smoothness"]["max_second_difference_norm"]
    )


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


def build_mid360_like_observed_point_cloud(
    obstacle_geometry_config: list[dict[str, Any]],
    robot_pose: np.ndarray,
    sensor_config: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Raycast axis-aligned box geometry into a world-frame observed cloud."""

    sensor = _build_sensor_config({} if sensor_config is None else sensor_config)
    if not obstacle_geometry_config:
        return np.empty((0, 3), dtype=np.float32)

    origin = np.asarray(robot_pose, dtype=np.float32).reshape(-1)[:3]
    directions = _mid360_like_world_ray_directions(robot_pose, sensor)
    hits = []
    for direction in directions:
        hit = _nearest_box_raycast_hit(
            origin,
            direction,
            obstacle_geometry_config,
            min_range_m=float(sensor["min_range_m"]),
            max_range_m=float(sensor["max_range_m"]),
        )
        if hit is not None:
            hits.append(hit)

    if not hits:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(hits, dtype=np.float32).reshape((-1, 3))


def _build_observed_point_cloud_for_step(
    *,
    sensor_config: dict[str, Any] | None,
    obstacle_geometry_config: list[dict[str, Any]],
    robot_pose: np.ndarray,
) -> np.ndarray:
    return build_mid360_like_observed_point_cloud(
        obstacle_geometry_config,
        robot_pose,
        sensor_config,
    )


def _controller_point_budget(
    controller_config: Mapping[str, Any],
) -> int:
    if "max_obs_num" in controller_config:
        point_budget = int(controller_config["max_obs_num"])
    else:
        raise ValueError(
            "MID-360-like 3D scenario config requires controller.max_obs_num "
            "for the controller obstacle point budget."
        )
    if point_budget <= 0:
        raise ValueError("controller.max_obs_num must be positive.")
    return point_budget


def _validate_observation_config(
    *,
    sensor_config: dict[str, Any] | None,
) -> None:
    if sensor_config is not None:
        return
    raise ValueError(
        "3D scenario config requires a top-level sensor section for the "
        "MID-360-like observation path."
    )


def _build_sensor_config(config: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if config is None:
        return None
    sensor = dict(config)
    sensor_type = str(sensor.get("type", "mid360_like"))
    if sensor_type != "mid360_like":
        raise ValueError(f"Unsupported 3D sensor type: {sensor_type}")

    normalized = {
        "type": "mid360_like",
        "min_range_m": float(sensor.get("min_range_m", 0.1)),
        "max_range_m": float(sensor.get("max_range_m", 10.0)),
        "horizontal_fov_deg": float(sensor.get("horizontal_fov_deg", 360.0)),
        "vertical_min_deg": float(sensor.get("vertical_min_deg", -7.0)),
        "vertical_max_deg": float(sensor.get("vertical_max_deg", 52.0)),
        "horizontal_samples": int(
            sensor.get("horizontal_samples", sensor.get("horizontal_sample_count", 180))
        ),
        "vertical_samples": int(
            sensor.get("vertical_samples", sensor.get("vertical_sample_count", 32))
        ),
        "noise_std_m": float(sensor.get("noise_std_m", 0.0)),
        "dropout_probability": float(sensor.get("dropout_probability", 0.0)),
        "seed": sensor.get("seed"),
    }
    if normalized["min_range_m"] < 0.0:
        raise ValueError("sensor.min_range_m must be non-negative.")
    if normalized["max_range_m"] <= normalized["min_range_m"]:
        raise ValueError("sensor.max_range_m must be greater than min_range_m.")
    if not 0.0 < normalized["horizontal_fov_deg"] <= 360.0:
        raise ValueError("sensor.horizontal_fov_deg must be in (0, 360].")
    if normalized["vertical_max_deg"] < normalized["vertical_min_deg"]:
        raise ValueError("sensor.vertical_max_deg must be >= vertical_min_deg.")
    if normalized["horizontal_samples"] <= 0:
        raise ValueError("sensor.horizontal_samples must be positive.")
    if normalized["vertical_samples"] <= 0:
        raise ValueError("sensor.vertical_samples must be positive.")
    if normalized["noise_std_m"] != 0.0:
        raise ValueError("MID-360-like sensor noise is reserved but not implemented.")
    if normalized["dropout_probability"] != 0.0:
        raise ValueError("MID-360-like sensor dropout is reserved but not implemented.")
    return normalized


def _mid360_like_world_ray_directions(
    robot_pose: np.ndarray,
    sensor_config: Mapping[str, Any],
) -> np.ndarray:
    horizontal_angles = _sample_horizontal_angles(
        fov_deg=float(sensor_config["horizontal_fov_deg"]),
        sample_count=int(sensor_config["horizontal_samples"]),
    )
    vertical_angles = _sample_vertical_angles(
        vertical_min_deg=float(sensor_config["vertical_min_deg"]),
        vertical_max_deg=float(sensor_config["vertical_max_deg"]),
        sample_count=int(sensor_config["vertical_samples"]),
    )

    local_directions = []
    for vertical_angle in vertical_angles:
        cos_elevation = np.cos(vertical_angle)
        for horizontal_angle in horizontal_angles:
            local_directions.append(
                [
                    cos_elevation * np.cos(horizontal_angle),
                    cos_elevation * np.sin(horizontal_angle),
                    np.sin(vertical_angle),
                ]
            )
    local_directions = np.asarray(local_directions, dtype=np.float32)

    pose = np.asarray(robot_pose, dtype=np.float32).reshape(-1)
    c = np.cos(pose[3])
    s = np.sin(pose[3])
    rotation_local_to_global = np.array([[c, s], [-s, c]], dtype=np.float32)
    world_directions = local_directions.copy()
    world_directions[:, :2] = world_directions[:, :2] @ rotation_local_to_global
    return world_directions.astype(np.float32)


def _sample_horizontal_angles(*, fov_deg: float, sample_count: int) -> np.ndarray:
    if sample_count == 1:
        return np.asarray([0.0], dtype=np.float32)
    fov = np.deg2rad(fov_deg)
    if np.isclose(fov_deg, 360.0):
        return np.linspace(0.0, fov, sample_count, endpoint=False, dtype=np.float32)
    return np.linspace(-0.5 * fov, 0.5 * fov, sample_count, dtype=np.float32)


def _sample_vertical_angles(
    *,
    vertical_min_deg: float,
    vertical_max_deg: float,
    sample_count: int,
) -> np.ndarray:
    if sample_count == 1:
        midpoint = 0.5 * (vertical_min_deg + vertical_max_deg)
        return np.asarray([np.deg2rad(midpoint)], dtype=np.float32)
    return np.linspace(
        np.deg2rad(vertical_min_deg),
        np.deg2rad(vertical_max_deg),
        sample_count,
        dtype=np.float32,
    )


def _nearest_box_raycast_hit(
    origin: np.ndarray,
    direction: np.ndarray,
    obstacle_geometry_config: list[dict[str, Any]],
    *,
    min_range_m: float,
    max_range_m: float,
) -> np.ndarray | None:
    nearest_t = np.inf
    nearest_hit = None
    for obstacle in obstacle_geometry_config:
        hit_t = _ray_axis_aligned_box_intersection(
            origin,
            direction,
            obstacle,
        )
        if hit_t is None or hit_t < min_range_m or hit_t > max_range_m:
            continue
        if hit_t < nearest_t:
            nearest_t = hit_t
            nearest_hit = origin + direction * hit_t
    if nearest_hit is None:
        return None
    return nearest_hit.astype(np.float32)


def _ray_axis_aligned_box_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    obstacle: Mapping[str, Any],
) -> float | None:
    center = np.asarray(obstacle["center"], dtype=np.float32)
    size = np.asarray(obstacle["size"], dtype=np.float32)
    half_size = size * 0.5
    bounds_min = center - half_size
    bounds_max = center + half_size

    t_min = -np.inf
    t_max = np.inf
    for axis in range(3):
        ray_component = float(direction[axis])
        if abs(ray_component) < 1e-8:
            if origin[axis] < bounds_min[axis] or origin[axis] > bounds_max[axis]:
                return None
            continue
        inv_d = 1.0 / ray_component
        t1 = (bounds_min[axis] - origin[axis]) * inv_d
        t2 = (bounds_max[axis] - origin[axis]) * inv_d
        t_near = min(t1, t2)
        t_far = max(t1, t2)
        t_min = max(t_min, t_near)
        t_max = min(t_max, t_far)
        if t_min > t_max:
            return None

    if t_min < 0.0:
        return None
    return float(t_min)


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
    *,
    reference_window_lookahead_distance: float | None = None,
) -> np.ndarray:
    return transfer_from_global_to_local_frame(
        select_global_plan(
            global_reference_path,
            robot_pose,
            time_steps,
            reference_window_lookahead_distance=reference_window_lookahead_distance,
        ),
        robot_pose,
    )


def select_global_plan(
    global_reference_path: np.ndarray,
    robot_pose: np.ndarray,
    time_steps: int,
    *,
    reference_window_lookahead_distance: float | None = None,
) -> np.ndarray:
    distances = np.linalg.norm(global_reference_path[:, :3] - robot_pose[:3], axis=1)
    nearest_idx = int(np.argmin(distances))
    if reference_window_lookahead_distance is not None:
        return _resample_reference_window(
            global_reference_path,
            nearest_idx=nearest_idx,
            time_steps=time_steps,
            lookahead_distance=reference_window_lookahead_distance,
        )
    end_idx = nearest_idx + time_steps
    if end_idx <= global_reference_path.shape[0]:
        global_plan = global_reference_path[nearest_idx:end_idx]
    else:
        pad_count = end_idx - global_reference_path.shape[0]
        padding = np.repeat(global_reference_path[-1][None, :], pad_count, axis=0)
        global_plan = np.vstack([global_reference_path[nearest_idx:], padding])
    return global_plan.astype(np.float32)


def _resample_reference_window(
    global_reference_path: np.ndarray,
    *,
    nearest_idx: int,
    time_steps: int,
    lookahead_distance: float,
) -> np.ndarray:
    if time_steps <= 0:
        raise ValueError("time_steps must be positive.")
    if lookahead_distance <= 0.0:
        raise ValueError("reference_window_lookahead_distance must be positive.")

    reference_path = np.asarray(global_reference_path, dtype=np.float32)
    suffix = reference_path[nearest_idx:]
    if suffix.shape[0] == 0:
        suffix = reference_path[-1:]
    if time_steps == 1 or suffix.shape[0] == 1:
        return np.repeat(suffix[0][None, :], time_steps, axis=0).astype(np.float32)

    segment_lengths = np.linalg.norm(np.diff(suffix[:, :3], axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    sample_distances = np.linspace(0.0, lookahead_distance, time_steps)

    window = np.empty((time_steps, reference_path.shape[1]), dtype=np.float32)
    for sample_index, sample_distance in enumerate(sample_distances):
        if sample_distance >= cumulative[-1]:
            window[sample_index] = suffix[-1]
            continue
        segment_index = int(np.searchsorted(cumulative, sample_distance, side="right") - 1)
        segment_index = min(segment_index, suffix.shape[0] - 2)
        segment_length = segment_lengths[segment_index]
        if segment_length <= 1e-8:
            alpha = 0.0
        else:
            alpha = (sample_distance - cumulative[segment_index]) / segment_length
        window[sample_index] = _interpolate_reference_state(
            suffix[segment_index],
            suffix[segment_index + 1],
            float(alpha),
        )
    return window.astype(np.float32)


def _interpolate_reference_state(
    start: np.ndarray,
    end: np.ndarray,
    alpha: float,
) -> np.ndarray:
    interpolated = start + (end - start) * alpha
    if interpolated.shape[0] >= 4:
        yaw_delta = np.arctan2(np.sin(end[3] - start[3]), np.cos(end[3] - start[3]))
        interpolated[3] = start[3] + yaw_delta * alpha
    return interpolated.astype(np.float32)


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


def _sample_global_rollouts(
    local_rollouts: np.ndarray | None,
    pose: np.ndarray,
    *,
    max_rollouts: int,
) -> np.ndarray:
    if local_rollouts is None or max_rollouts <= 0:
        return np.empty((0, 0, 4), dtype=np.float32)

    rollouts = np.asarray(local_rollouts, dtype=np.float32)
    if rollouts.size == 0:
        return np.empty((0, 0, 4), dtype=np.float32)
    rollouts = rollouts.reshape((rollouts.shape[0], rollouts.shape[1], 4))
    sample_count = min(max_rollouts, rollouts.shape[0])
    sample_indices = np.linspace(
        0,
        rollouts.shape[0] - 1,
        sample_count,
        dtype=np.int32,
    )
    sampled = rollouts[sample_indices]
    return transfer_from_local_to_global_frame(sampled, pose).astype(np.float32)


def _bounded_rollouts_for_frame(
    rollout_history: list[np.ndarray],
    frame_index: int,
    *,
    max_rollouts: int,
) -> list[Any]:
    if max_rollouts <= 0 or frame_index >= len(rollout_history):
        return []
    return rollout_history[frame_index][:max_rollouts].tolist()


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
    parser.add_argument(
        "--replay-dir",
        type=Path,
        help="Write one Offline Web replay JSON per scenario when using --all-static.",
    )
    parser.add_argument(
        "--replay-rollouts",
        action="store_true",
        help="Include bounded sampled MPPI rollouts in replay JSON.",
    )
    parser.add_argument(
        "--replay-max-rollouts",
        type=int,
        default=8,
        help="Maximum sampled rollouts per replay frame when rollouts are enabled.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.all_static and args.replay_json is not None:
        parser.error("--replay-json is only supported for a selected scenario.")
    if not args.all_static and args.replay_dir is not None:
        parser.error("--replay-dir requires --all-static.")

    if args.all_static:
        results = run_3d_static_scenario_suite(
            collect_rollouts=args.replay_rollouts,
            max_rollouts=args.replay_max_rollouts,
        )
        replay_artifacts = None
        if args.replay_dir is not None:
            replay_artifacts = write_3d_baseline_replay_artifacts(
                results,
                args.replay_dir,
                include_rollouts=args.replay_rollouts,
                max_rollouts=args.replay_max_rollouts,
            )
        baseline_report = build_3d_baseline_report(
            results,
            replay_artifacts=replay_artifacts,
        )
        summary_json = json.dumps(baseline_report, allow_nan=False, sort_keys=True)
        exit_ok = True
    else:
        config = (
            load_scenario_config(args.config)
            if args.config is not None
            else load_builtin_scenario_config(args.scenario)
        )
        result = run_3d_scenario(
            config,
            collect_rollouts=args.replay_rollouts,
            max_rollouts=args.replay_max_rollouts,
        )
        summary_json = json.dumps(result.summary, allow_nan=False, sort_keys=True)
        exit_ok = result.reached_goal and not result.collided
        if args.replay_json is not None:
            write_3d_replay_json(
                result,
                args.replay_json,
                include_rollouts=args.replay_rollouts,
                max_rollouts=args.replay_max_rollouts,
            )

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
    controller_point_budget: int,
    robot_volume_config: list[dict[str, Any]],
    clearance_margin: float,
) -> MPPIController3D:
    kwargs = copy.deepcopy(dict(controller_config))
    kwargs.setdefault("model_dt", model_dt)
    kwargs.setdefault("time_steps", 12)
    kwargs.setdefault("max_obs_num", controller_point_budget)
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


def _build_obstacle_geometry_config(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    geometry = config.get("geometry", [])
    if geometry is None:
        return []
    result = []
    for index, obstacle in enumerate(geometry):
        obstacle_type = str(obstacle.get("type", "box"))
        if obstacle_type != "box":
            raise ValueError(
                f"Unsupported 3D obstacle geometry type at index {index}: "
                f"{obstacle_type}"
            )
        center = _finite_vector3(
            obstacle.get("center"),
            f"obstacle geometry {index} center",
        )
        size = _finite_vector3(
            obstacle.get("size"),
            f"obstacle geometry {index} size",
        )
        if any(value <= 0.0 for value in size):
            raise ValueError(f"obstacle geometry {index} size values must be positive.")
        result.append(
            {
                "type": "box",
                "center": center,
                "size": size,
            }
        )
    return result


def _finite_vector3(value: Any, name: str) -> list[float]:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,).")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    return [float(item) for item in vector.tolist()]


def _minimum_state_clearance(
    robot_volume: BoxUnionVolume3D,
    robot_volume_config: list[dict[str, Any]],
    obstacle_geometry_config: list[dict[str, Any]],
    global_obstacle_points: np.ndarray,
    robot_pose: np.ndarray,
) -> float | None:
    if obstacle_geometry_config:
        return _minimum_geometry_clearance(
            robot_volume_config,
            obstacle_geometry_config,
            robot_pose,
        )
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


def _minimum_geometry_clearance(
    robot_volume_config: list[dict[str, Any]],
    obstacle_geometry_config: list[dict[str, Any]],
    robot_pose: np.ndarray,
) -> float | None:
    robot_boxes = _robot_world_boxes(robot_volume_config, robot_pose)
    obstacle_boxes = [
        _axis_aligned_world_box(obstacle)
        for obstacle in obstacle_geometry_config
    ]
    clearances = [
        _signed_obb_clearance(robot_box, obstacle_box)
        for robot_box in robot_boxes
        for obstacle_box in obstacle_boxes
    ]
    if not clearances:
        return None
    return float(min(clearances))


def _robot_world_boxes(
    robot_volume_config: list[dict[str, Any]],
    robot_pose: np.ndarray,
) -> list[dict[str, np.ndarray]]:
    pose = np.asarray(robot_pose, dtype=np.float32).reshape(-1)
    c = float(np.cos(pose[3]))
    s = float(np.sin(pose[3]))
    axes = np.asarray(
        [
            [c, s, 0.0],
            [-s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    boxes = []
    for box in robot_volume_config:
        local_center = np.asarray(box.get("center", [0.0, 0.0, 0.0]), dtype=np.float32)
        size = np.asarray(box["size"], dtype=np.float32)
        center = transfer_from_local_to_global_frame(local_center[None, :], pose)[0]
        boxes.append(
            {
                "center": center.astype(np.float32),
                "axes": axes,
                "half_size": (size * 0.5).astype(np.float32),
            }
        )
    return boxes


def _axis_aligned_world_box(obstacle: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        "center": np.asarray(obstacle["center"], dtype=np.float32),
        "axes": np.eye(3, dtype=np.float32),
        "half_size": (np.asarray(obstacle["size"], dtype=np.float32) * 0.5).astype(
            np.float32
        ),
    }


def _signed_obb_clearance(
    first: Mapping[str, np.ndarray],
    second: Mapping[str, np.ndarray],
) -> float:
    intersects, penetration = _obb_intersection_penetration(first, second)
    if intersects:
        return -float(penetration)
    return float(_obb_surface_distance(first, second))


def _obb_intersection_penetration(
    first: Mapping[str, np.ndarray],
    second: Mapping[str, np.ndarray],
) -> tuple[bool, float]:
    axes = []
    first_axes = np.asarray(first["axes"], dtype=np.float32)
    second_axes = np.asarray(second["axes"], dtype=np.float32)
    axes.extend(first_axes)
    axes.extend(second_axes)
    for first_axis in first_axes:
        for second_axis in second_axes:
            cross_axis = np.cross(first_axis, second_axis)
            if np.linalg.norm(cross_axis) > 1e-7:
                axes.append(cross_axis)

    min_overlap = np.inf
    for axis in axes:
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-7:
            continue
        unit_axis = np.asarray(axis, dtype=np.float32) / norm
        first_min, first_max = _project_obb(first, unit_axis)
        second_min, second_max = _project_obb(second, unit_axis)
        overlap = min(first_max, second_max) - max(first_min, second_min)
        if overlap < 0.0:
            return False, 0.0
        min_overlap = min(min_overlap, overlap)
    if not np.isfinite(min_overlap):
        min_overlap = 0.0
    return True, float(min_overlap)


def _project_obb(
    box: Mapping[str, np.ndarray],
    axis: np.ndarray,
) -> tuple[float, float]:
    center = float(np.dot(box["center"], axis))
    radius = float(
        np.sum(np.asarray(box["half_size"]) * np.abs(np.asarray(box["axes"]) @ axis))
    )
    return center - radius, center + radius


def _obb_surface_distance(
    first: Mapping[str, np.ndarray],
    second: Mapping[str, np.ndarray],
) -> float:
    distances = []
    first_corners = _obb_corners(first)
    second_corners = _obb_corners(second)
    distances.extend(_point_to_obb_distance(corner, second) for corner in first_corners)
    distances.extend(_point_to_obb_distance(corner, first) for corner in second_corners)

    first_edges = _obb_edges(first_corners)
    second_edges = _obb_edges(second_corners)
    for first_start, first_end in first_edges:
        for second_start, second_end in second_edges:
            distances.append(
                _segment_segment_distance(
                    first_start,
                    first_end,
                    second_start,
                    second_end,
                )
            )
    return float(min(distances)) if distances else np.inf


def _obb_corners(box: Mapping[str, np.ndarray]) -> list[np.ndarray]:
    center = np.asarray(box["center"], dtype=np.float32)
    axes = np.asarray(box["axes"], dtype=np.float32)
    half_size = np.asarray(box["half_size"], dtype=np.float32)
    corners = []
    for x_sign in (-1.0, 1.0):
        for y_sign in (-1.0, 1.0):
            for z_sign in (-1.0, 1.0):
                signs = np.asarray([x_sign, y_sign, z_sign], dtype=np.float32)
                corners.append(center + (signs * half_size) @ axes)
    return corners


def _obb_edges(corners: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    edges = []
    signs = [
        (x_sign, y_sign, z_sign)
        for x_sign in (-1.0, 1.0)
        for y_sign in (-1.0, 1.0)
        for z_sign in (-1.0, 1.0)
    ]
    for first_index, first_signs in enumerate(signs):
        for second_index in range(first_index + 1, len(signs)):
            second_signs = signs[second_index]
            if sum(a != b for a, b in zip(first_signs, second_signs)) == 1:
                edges.append((corners[first_index], corners[second_index]))
    return edges


def _point_to_obb_distance(point: np.ndarray, box: Mapping[str, np.ndarray]) -> float:
    delta = np.asarray(point, dtype=np.float32) - np.asarray(
        box["center"],
        dtype=np.float32,
    )
    local = np.asarray(box["axes"], dtype=np.float32) @ delta
    outside = np.maximum(
        np.abs(local) - np.asarray(box["half_size"], dtype=np.float32),
        0.0,
    )
    return float(np.linalg.norm(outside))


def _segment_segment_distance(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> float:
    u = np.asarray(first_end, dtype=np.float64) - np.asarray(
        first_start,
        dtype=np.float64,
    )
    v = np.asarray(second_end, dtype=np.float64) - np.asarray(
        second_start,
        dtype=np.float64,
    )
    w = np.asarray(first_start, dtype=np.float64) - np.asarray(
        second_start,
        dtype=np.float64,
    )
    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))
    denominator = a * c - b * b
    small = 1e-12

    if a <= small and c <= small:
        return float(np.linalg.norm(w))
    if a <= small:
        t = np.clip(e / c, 0.0, 1.0)
        closest = w - t * v
        return float(np.linalg.norm(closest))
    if c <= small:
        s = np.clip(-d / a, 0.0, 1.0)
        closest = w + s * u
        return float(np.linalg.norm(closest))

    if denominator < small:
        s = 0.0
    else:
        s = np.clip((b * e - c * d) / denominator, 0.0, 1.0)
    t = (b * s + e) / c
    if t < 0.0:
        t = 0.0
        s = np.clip(-d / a, 0.0, 1.0)
    elif t > 1.0:
        t = 1.0
        s = np.clip((b - d) / a, 0.0, 1.0)

    closest = w + s * u - t * v
    return float(np.linalg.norm(closest))


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
