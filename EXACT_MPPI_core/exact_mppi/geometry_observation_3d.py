"""3D geometry observation for yaw-only scenario simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class GeometryObservation3D:
    observed_point_cloud: np.ndarray
    local_observation_points: np.ndarray
    clearance: float | None


def observe_3d_geometry(
    *,
    obstacle_geometry_config: list[dict[str, Any]],
    robot_volume_config: list[dict[str, Any]],
    robot_pose: np.ndarray,
    sensor_config: Mapping[str, Any],
) -> GeometryObservation3D:
    observed_point_cloud = build_mid360_like_observed_point_cloud(
        obstacle_geometry_config,
        robot_pose,
        sensor_config,
    )
    local_observation_points = _world_points_to_robot_local_yaw_frame(
        observed_point_cloud,
        robot_pose,
    ).astype(np.float32)
    clearance = minimum_state_clearance(
        robot_volume_config,
        obstacle_geometry_config,
        robot_pose,
    )
    return GeometryObservation3D(
        observed_point_cloud=observed_point_cloud,
        local_observation_points=local_observation_points,
        clearance=clearance,
    )


def build_mid360_like_observed_point_cloud(
    obstacle_geometry_config: list[dict[str, Any]],
    robot_pose: np.ndarray,
    sensor_config: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Raycast axis-aligned box geometry into a world-frame observed cloud."""

    sensor = build_mid360_like_sensor_config(
        {} if sensor_config is None else sensor_config
    )
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


def build_mid360_like_sensor_config(
    config: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
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


def minimum_state_clearance(
    robot_volume_config: list[dict[str, Any]],
    obstacle_geometry_config: list[dict[str, Any]],
    robot_pose: np.ndarray,
) -> float | None:
    if obstacle_geometry_config:
        return minimum_geometry_clearance(
            robot_volume_config,
            obstacle_geometry_config,
            robot_pose,
        )
    return None


def minimum_geometry_clearance(
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


def _world_points_to_robot_local_yaw_frame(
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
    return out


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
        center = _robot_local_point_to_world(local_center, pose)
        boxes.append(
            {
                "center": center.astype(np.float32),
                "axes": axes,
                "half_size": (size * 0.5).astype(np.float32),
            }
        )
    return boxes


def _robot_local_point_to_world(point: np.ndarray, pose: np.ndarray) -> np.ndarray:
    p = np.asarray(point, dtype=np.float32).copy()
    pose = np.asarray(pose, dtype=np.float32).reshape(-1)
    c = np.cos(pose[3])
    s = np.sin(pose[3])
    rotation_local_to_global = np.array([[c, s], [-s, c]], dtype=np.float32)
    p[:2] = p[:2] @ rotation_local_to_global + pose[:2]
    p[2] = p[2] + pose[2]
    return p


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
