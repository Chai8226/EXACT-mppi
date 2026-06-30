import argparse
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from exact_mppi.mppi_3d import BoxUnionVolume3D, MPPIController3D


@dataclass(frozen=True)
class ExampleResult3D:
    reached_goal: bool
    collided: bool
    min_sdf_clearance: float
    final_state: np.ndarray
    state_history: np.ndarray
    command_history: np.ndarray
    global_reference_path: np.ndarray
    global_obstacle_points: np.ndarray


def _wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


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


def build_global_obstacle_points() -> np.ndarray:
    xs = np.linspace(1.7, 2.25, 4, dtype=np.float32)
    ys = np.linspace(-0.55, 0.55, 5, dtype=np.float32)
    zs = np.linspace(-0.2, 0.75, 5, dtype=np.float32)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)
    return grid.reshape(-1, 3).astype(np.float32)


def build_global_reference_path(point_count: int = 96) -> np.ndarray:
    waypoints = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.8, 0.0, 0.0, 0.0],
            [1.45, 0.0, 1.2, 0.0],
            [2.55, 0.0, 1.2, 0.0],
            [3.05, 0.0, 0.15, 0.0],
            [3.45, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    segment_lengths = np.linalg.norm(np.diff(waypoints[:, :3], axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    samples = np.linspace(0.0, cumulative[-1], point_count, dtype=np.float32)

    path = np.empty((point_count, 4), dtype=np.float32)
    for dim in range(4):
        path[:, dim] = np.interp(samples, cumulative, waypoints[:, dim])
    return path


def build_range_based_local_observation(
    global_obstacle_points: np.ndarray,
    robot_pose: np.ndarray,
    observation_range: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    deltas = global_obstacle_points - robot_pose[:3]
    distances = np.linalg.norm(deltas, axis=1)
    selected_indices = np.flatnonzero(distances <= observation_range)
    if selected_indices.size > max_points:
        nearest_order = np.argsort(distances[selected_indices])[:max_points]
        selected_indices = selected_indices[nearest_order]

    packed = np.zeros((max_points, 3), dtype=np.float32)
    mask = np.zeros((max_points,), dtype=np.float32)
    if selected_indices.size:
        local_points = transfer_from_global_to_local_frame(
            global_obstacle_points[selected_indices],
            robot_pose,
        )
        packed[: selected_indices.size] = local_points
        mask[: selected_indices.size] = 1.0
    return packed, mask


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


def minimum_state_clearance(
    robot_volume: BoxUnionVolume3D,
    global_obstacle_points: np.ndarray,
    robot_pose: np.ndarray,
) -> float:
    body_points = transfer_from_global_to_local_frame(
        global_obstacle_points,
        robot_pose,
    )
    distances = robot_volume.signed_distance(jnp.asarray(body_points, dtype=jnp.float32))
    return float(jax.device_get(jnp.min(distances)))


def run_3d_obstacle_avoidance_example(
    max_steps: int = 80,
    goal_tolerance: float = 0.28,
    clearance_margin: float = 0.04,
) -> ExampleResult3D:
    model_dt = 0.15
    time_steps = 12
    max_obstacle_points = 48
    robot_volume_config = [
        {"center": [0.0, 0.0, 0.0], "size": [0.35, 0.35, 0.35]},
    ]
    robot_volume = BoxUnionVolume3D.from_config(robot_volume_config)
    controller = MPPIController3D(
        model_dt=model_dt,
        time_steps=time_steps,
        batch_size=48,
        iteration_count=1,
        seed=23,
        max_obs_num=max_obstacle_points,
        vx_max=1.15,
        vx_min=-0.2,
        vy_max=0.45,
        vz_max=1.0,
        wz_max=1.0,
        vx_std=0.25,
        vy_std=0.18,
        vz_std=0.24,
        wz_std=0.12,
        temperature=0.25,
        goal_weight=6.0,
        path_weight=7.0,
        robot_volume_config=robot_volume_config,
        obstacles_repulsion_weight=0.8,
        obstacles_critical_weight=35.0,
        obstacles_collision_margin_distance=clearance_margin,
        obstacles_repulsion_distance=0.55,
        PreferForwardCritic={"enabled": True, "cost_weight": 0.5},
        VelocityDeadbandCritic={"enabled": False},
        TwirlingCritic={"enabled": True, "cost_weight": 0.1},
        TrajectoryValidator={
            "collision_lookahead_time": 1.0,
            "collision_margin_distance": clearance_margin,
        },
    )

    obstacle_points = build_global_obstacle_points()
    reference_path = build_global_reference_path()
    goal = reference_path[-1].copy()

    state = reference_path[0].copy()
    speed = np.zeros(4, dtype=np.float32)
    state_history = [state.copy()]
    command_history = []
    min_clearance = minimum_state_clearance(robot_volume, obstacle_points, state)

    for _ in range(max_steps):
        local_plan = select_local_plan(reference_path, state, time_steps)
        local_goal = transfer_from_global_to_local_frame(goal[None, :], state)[0]
        local_obstacles, local_obstacle_mask = build_range_based_local_observation(
            obstacle_points,
            state,
            observation_range=1.7,
            max_points=max_obstacle_points,
        )
        valid_local_obstacles = local_obstacles[local_obstacle_mask > 0.0]

        command = controller.computeVelocityCommands(
            robot_pose=np.zeros(4, dtype=np.float32),
            robot_speed=speed,
            plan=local_plan,
            goal=local_goal,
            obstacle_points=valid_local_obstacles,
        )
        command = np.asarray(command, dtype=np.float32)
        state = integrate_yaw_only_3d_state(state, command, model_dt)
        speed = command

        clearance = minimum_state_clearance(robot_volume, obstacle_points, state)
        min_clearance = min(min_clearance, clearance)
        state_history.append(state.copy())
        command_history.append(command.copy())

        if np.linalg.norm(state[:3] - goal[:3]) <= goal_tolerance:
            break

    final_distance = float(np.linalg.norm(state[:3] - goal[:3]))
    reached_goal = final_distance <= goal_tolerance
    collided = min_clearance < clearance_margin
    return ExampleResult3D(
        reached_goal=reached_goal,
        collided=collided,
        min_sdf_clearance=float(min_clearance),
        final_state=state.copy(),
        state_history=np.asarray(state_history, dtype=np.float32),
        command_history=np.asarray(command_history, dtype=np.float32),
        global_reference_path=reference_path,
        global_obstacle_points=obstacle_points,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a Core-only yaw-only 3D MPPI obstacle-avoidance example."
    )
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--goal-tolerance", type=float, default=0.28)
    parser.add_argument("--clearance-margin", type=float, default=0.04)
    args = parser.parse_args()

    result = run_3d_obstacle_avoidance_example(
        max_steps=args.max_steps,
        goal_tolerance=args.goal_tolerance,
        clearance_margin=args.clearance_margin,
    )
    final_distance = np.linalg.norm(
        result.final_state[:3] - result.global_reference_path[-1, :3]
    )
    print(f"Reached goal: {result.reached_goal}")
    print(f"Collided: {result.collided}")
    print(f"Final xyz distance to goal: {final_distance:.3f}")
    print(f"Minimum SDF clearance: {result.min_sdf_clearance:.3f}")

    if not result.reached_goal:
        print("Failure: 3D MPPI example missed the goal.")
        return 1
    if result.collided:
        print("Failure: 3D MPPI example violated the SDF clearance margin.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
