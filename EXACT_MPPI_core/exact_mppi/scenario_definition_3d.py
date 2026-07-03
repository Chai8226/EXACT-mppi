"""3D scenario definition loading for yaw-only scenario simulation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from exact_mppi.geometry_observation_3d import build_mid360_like_sensor_config


DEFAULT_REFERENCE_WINDOW_LOOKAHEAD_DISTANCE = 1.2


@dataclass(frozen=True, slots=True)
class ScenarioDefinition3D:
    name: str
    model_dt: float
    max_steps: int
    goal_tolerance: float
    clearance_margin: float
    reference_window_lookahead_distance: float
    reference_path: np.ndarray
    robot_volume_config: list[dict[str, Any]]
    obstacle_geometry_config: list[dict[str, Any]]
    sensor_config: dict[str, Any]
    controller_config: dict[str, Any]
    controller_point_budget: int


def build_3d_scenario_definition(config: Mapping[str, Any]) -> ScenarioDefinition3D:
    cfg = copy.deepcopy(dict(config))
    simulation = dict(cfg.get("simulation", {}))
    obstacle_config = cfg.get("obstacles", {})
    sensor_config = build_3d_sensor_config(cfg.get("sensor"))
    controller_config = dict(cfg.get("controller", {}))

    return ScenarioDefinition3D(
        name=str(cfg["name"]),
        model_dt=float(simulation.get("model_dt", 0.15)),
        max_steps=int(simulation.get("max_steps", 80)),
        goal_tolerance=float(simulation.get("goal_tolerance", 0.28)),
        clearance_margin=float(simulation.get("clearance_margin", 0.04)),
        reference_window_lookahead_distance=float(
            simulation.get(
                "reference_window_lookahead_distance",
                DEFAULT_REFERENCE_WINDOW_LOOKAHEAD_DISTANCE,
            )
        ),
        reference_path=build_3d_reference_path(cfg["reference_path"]),
        robot_volume_config=load_3d_robot_volume_config(cfg),
        obstacle_geometry_config=build_3d_obstacle_geometry_config(obstacle_config),
        sensor_config=sensor_config,
        controller_config=controller_config,
        controller_point_budget=controller_point_budget(controller_config),
    )


def build_3d_sensor_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    sensor_config = build_mid360_like_sensor_config(config)
    if sensor_config is not None:
        return sensor_config
    raise ValueError(
        "3D scenario config requires a top-level sensor section for the "
        "MID-360-like observation path."
    )


def controller_point_budget(
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


def load_3d_robot_volume_config(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    robot_volume = config.get("robot_volume", {})
    if isinstance(robot_volume, Mapping):
        volume_type = robot_volume.get("type", "box_union")
        if volume_type != "box_union":
            raise ValueError(f"Unsupported 3D robot volume type: {volume_type}")
        boxes = robot_volume.get("boxes", robot_volume)
    else:
        boxes = robot_volume
    if not isinstance(boxes, list) or not boxes:
        raise ValueError("3D scenario config requires at least one robot volume box.")
    return [dict(box) for box in boxes]


def build_3d_reference_path(config: Mapping[str, Any]) -> np.ndarray:
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


def build_3d_obstacle_geometry_config(
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if "points" in config:
        raise ValueError(
            "3D scenario config no longer supports obstacles.points; "
            "use obstacles.geometry."
        )
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
        center = finite_vector3(
            obstacle.get("center"),
            f"obstacle geometry {index} center",
        )
        size = finite_vector3(
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


def finite_vector3(value: Any, name: str) -> list[float]:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,).")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    return [float(item) for item in vector.tolist()]
