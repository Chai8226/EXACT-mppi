import json

import jax
import jax.numpy as jnp
import numpy as np

from exact_mppi.mppi_3d import BoxUnionVolume3D
from exact_mppi.scenario_runner_3d import (
    STATIC_3D_SCENARIOS,
    build_3d_replay_data,
    compute_3d_smoothness_telemetry,
    load_builtin_scenario_config,
    run_3d_scenario,
    run_3d_static_scenario_suite,
    transfer_from_global_to_local_frame,
    write_3d_replay_json,
)


def test_open_track_3d_runs_headlessly_from_builtin_config():
    config = load_builtin_scenario_config("open_track_3d")

    result = run_3d_scenario(config)
    summary = result.summary

    assert summary["scenario"] == "open_track_3d"
    assert summary["reached_goal"] is True
    assert summary["collided"] is False
    assert summary["step_count"] > 0
    assert summary["final_distance"] <= config["simulation"]["goal_tolerance"]
    assert "minimum_clearance" in summary
    assert "command_smoothness" in summary
    assert "trajectory_smoothness" in summary
    assert_finite_metric_values(summary["command_smoothness"])
    assert_finite_metric_values(summary["trajectory_smoothness"])
    json.dumps(summary, allow_nan=False)

    assert result.state_history.shape[1] == 4
    assert result.command_history.shape[1] == 4
    assert result.global_reference_path.shape[1] == 4
    assert result.global_obstacle_points.shape[1] == 3


def test_open_track_3d_builtin_config_is_deterministic():
    config = load_builtin_scenario_config("open_track_3d")

    first = run_3d_scenario(config)
    second = run_3d_scenario(config)

    assert first.summary == second.summary
    np.testing.assert_allclose(first.state_history, second.state_history)
    np.testing.assert_allclose(first.command_history, second.command_history)


def test_narrow_gap_t_volume_3d_runs_with_authoritative_robot_volume():
    config = load_builtin_scenario_config("narrow_gap_t_volume_3d")

    result = run_3d_scenario(config)
    summary = result.summary

    assert summary["scenario"] == "narrow_gap_t_volume_3d"
    assert summary["reached_goal"] is True
    assert summary["collided"] is False
    assert summary["step_count"] > 0
    assert summary["final_distance"] <= config["simulation"]["goal_tolerance"]
    assert summary["minimum_clearance"] is not None
    assert summary["minimum_clearance"] >= config["simulation"]["clearance_margin"]
    assert_finite_metric_values(summary["command_smoothness"])
    assert_finite_metric_values(summary["trajectory_smoothness"])
    json.dumps(summary, allow_nan=False)

    assert config["robot_volume"]["type"] == "box_union"
    assert result.robot_volume_config == config["robot_volume"]["boxes"]
    assert len(result.robot_volume_config) == 2
    assert result.global_obstacle_points.shape[0] > 0
    assert_t_volume_has_nonconvex_notch(result.robot_volume_config)


def test_narrow_gap_t_volume_export_recomputes_reported_clearance():
    config = load_builtin_scenario_config("narrow_gap_t_volume_3d")

    result = run_3d_scenario(config)

    assert result.minimum_clearance is not None
    np.testing.assert_allclose(
        recompute_minimum_clearance_from_result_volume(result),
        result.minimum_clearance,
        atol=1e-5,
    )


def test_open_track_3d_exports_world_frame_replay_data(tmp_path):
    result = run_3d_scenario(load_builtin_scenario_config("open_track_3d"))

    replay = build_3d_replay_data(result)
    replay_path = tmp_path / "open_track_3d.replay.json"
    write_3d_replay_json(result, replay_path)
    written_replay = json.loads(replay_path.read_text(encoding="utf-8"))

    assert written_replay == replay
    assert_replay_scene(replay, result)
    assert_replay_frames(replay, result, require_clearance=False)
    assert_replay_geometry_is_world_frame(replay)


def test_narrow_gap_t_volume_3d_exports_finite_replay_data(tmp_path):
    result = run_3d_scenario(load_builtin_scenario_config("narrow_gap_t_volume_3d"))

    replay = build_3d_replay_data(result)
    replay_path = tmp_path / "narrow_gap_t_volume_3d.replay.json"
    write_3d_replay_json(result, replay_path)

    assert json.loads(replay_path.read_text(encoding="utf-8")) == replay
    assert_replay_scene(replay, result)
    assert_replay_frames(replay, result, require_clearance=True)
    assert_replay_geometry_is_world_frame(replay)
    assert replay["scene"]["robot_volume"]["boxes"] == result.robot_volume_config


def test_replay_rollouts_are_omitted_by_default():
    result = run_3d_scenario(load_builtin_scenario_config("open_track_3d"))

    replay = build_3d_replay_data(result)

    assert "rollouts" not in replay["frames"][0]
    assert result.rollout_history == []


def test_replay_rollouts_are_optional_bounded_and_world_frame(tmp_path):
    result = run_3d_scenario(
        load_builtin_scenario_config("open_track_3d"),
        collect_rollouts=True,
        max_rollouts=3,
    )

    replay = build_3d_replay_data(result, include_rollouts=True, max_rollouts=2)
    replay_path = tmp_path / "open_track_3d_rollouts.replay.json"
    write_3d_replay_json(
        result,
        replay_path,
        include_rollouts=True,
        max_rollouts=2,
    )
    written_replay = json.loads(replay_path.read_text(encoding="utf-8"))

    assert written_replay == replay
    assert len(result.rollout_history) == result.step_count
    assert len(replay["frames"][0]["rollouts"]) <= 2
    assert len(replay["frames"][-1]["rollouts"]) <= 2
    assert_replay_rollouts_are_finite_and_world_frame(replay)
    json.dumps(replay, allow_nan=False)


def test_static_3d_scenario_suite_contains_first_static_scenarios():
    assert STATIC_3D_SCENARIOS == (
        "open_track_3d",
        "narrow_gap_t_volume_3d",
        "vertical_gate_3d",
        "t_shape_trap_3d",
        "cluttered_corridor_3d",
    )


def test_new_static_3d_scenarios_run_headlessly_with_standard_summaries():
    for scenario_name in (
        "vertical_gate_3d",
        "t_shape_trap_3d",
        "cluttered_corridor_3d",
    ):
        config = load_builtin_scenario_config(scenario_name)

        result = run_3d_scenario(config)

        assert_standard_summary(result.summary, scenario_name)
        assert result.global_obstacle_points.shape[0] > 0


def test_static_3d_scenario_suite_runs_selected_scenarios_in_order():
    results = run_3d_static_scenario_suite(
        ["vertical_gate_3d", "cluttered_corridor_3d"]
    )

    assert [result.scenario for result in results] == [
        "vertical_gate_3d",
        "cluttered_corridor_3d",
    ]
    for result in results:
        assert_standard_summary(result.summary, result.scenario)


def test_static_3d_scenario_suite_runs_all_static_scenarios():
    results = run_3d_static_scenario_suite()

    assert [result.scenario for result in results] == list(STATIC_3D_SCENARIOS)
    for result in results:
        assert_standard_summary(result.summary, result.scenario)


def test_static_3d_scenario_suite_configs_do_not_define_dynamic_obstacles():
    for scenario_name in STATIC_3D_SCENARIOS:
        config = load_builtin_scenario_config(scenario_name)

        assert "dynamic_obstacles" not in config
        assert "dynamic_obstacles" not in config.get("obstacles", {})


def test_3d_smoothness_telemetry_uses_known_command_and_state_histories():
    command_history = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    )
    state_history = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
            [6.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    telemetry = compute_3d_smoothness_telemetry(
        command_history=command_history,
        state_history=state_history,
    )

    assert telemetry["command_smoothness"] == {
        "sample_count": 2,
        "mean_delta_norm": 2.0,
        "rms_delta_norm": np.sqrt(5.0),
        "max_delta_norm": 3.0,
        "total_delta_norm": 4.0,
    }
    assert telemetry["trajectory_smoothness"] == {
        "sample_count": 2,
        "mean_second_difference_norm": 1.0,
        "rms_second_difference_norm": 1.0,
        "max_second_difference_norm": 1.0,
        "total_second_difference_norm": 2.0,
    }


def test_3d_smoothness_telemetry_unwraps_yaw_before_trajectory_differences():
    telemetry = compute_3d_smoothness_telemetry(
        command_history=np.empty((0, 4), dtype=np.float32),
        state_history=np.asarray(
            [
                [0.0, 0.0, 0.0, np.pi - 0.1],
                [0.0, 0.0, 0.0, -np.pi + 0.1],
                [0.0, 0.0, 0.0, -np.pi + 0.3],
            ],
            dtype=np.float32,
        ),
    )

    assert telemetry["command_smoothness"] == {
        "sample_count": 0,
        "mean_delta_norm": 0.0,
        "rms_delta_norm": 0.0,
        "max_delta_norm": 0.0,
        "total_delta_norm": 0.0,
    }
    assert telemetry["trajectory_smoothness"]["sample_count"] == 1
    np.testing.assert_allclose(
        telemetry["trajectory_smoothness"]["max_second_difference_norm"],
        0.0,
        atol=1e-6,
    )


def assert_finite_metric_values(metrics):
    for value in metrics.values():
        assert np.isfinite(value)


def assert_standard_summary(summary, scenario_name):
    assert summary["scenario"] == scenario_name
    assert isinstance(summary["reached_goal"], bool)
    assert isinstance(summary["collided"], bool)
    assert np.isfinite(summary["final_distance"])
    assert "minimum_clearance" in summary
    if summary["minimum_clearance"] is not None:
        assert np.isfinite(summary["minimum_clearance"])
    assert summary["step_count"] > 0
    assert_finite_metric_values(summary["command_smoothness"])
    assert_finite_metric_values(summary["trajectory_smoothness"])
    json.dumps(summary, allow_nan=False)


def assert_replay_scene(replay, result):
    scene = replay["scene"]

    assert replay["schema_version"] == 1
    assert scene["scenario"] == result.scenario
    assert scene["coordinate_conventions"] == {
        "frame": "world",
        "state": "[x, y, z, yaw]",
        "command": "[vx, vy, vz, wz]",
        "yaw_unit": "radians",
    }
    assert scene["reference_path"] == result.global_reference_path.tolist()
    assert scene["obstacle_points"] == result.global_obstacle_points.tolist()
    assert scene["robot_volume"]["type"] == "box_union"
    assert scene["robot_volume"]["boxes"] == result.robot_volume_config
    assert replay["summary"] == result.summary
    json.dumps(replay, allow_nan=False)


def assert_replay_frames(replay, result, *, require_clearance):
    frames = replay["frames"]

    assert len(frames) == result.step_count
    assert len(frames) == len(result.command_history)
    assert len(frames) == len(result.local_plan_history)
    assert len(frames) == len(result.optimal_trajectory_history)

    for frame_index, frame in enumerate(frames):
        assert frame["frame_index"] == frame_index
        assert frame["state"] == result.state_history[frame_index + 1].tolist()
        assert frame["executed_path"] == result.state_history[: frame_index + 2].tolist()
        assert frame["command"] == result.command_history[frame_index].tolist()
        assert frame["local_plan"] == result.local_plan_history[frame_index].tolist()
        assert (
            frame["optimal_trajectory"]
            == result.optimal_trajectory_history[frame_index].tolist()
        )
        assert np.isfinite(frame["goal_distance"])
        assert "smoothness_telemetry" in frame
        assert_finite_metric_values(frame["smoothness_telemetry"]["command_smoothness"])
        assert_finite_metric_values(
            frame["smoothness_telemetry"]["trajectory_smoothness"]
        )
        assert_frame_arrays_are_finite(frame)
        if require_clearance:
            assert np.isfinite(frame["clearance"])
        else:
            assert frame["clearance"] is None or np.isfinite(frame["clearance"])


def assert_replay_geometry_is_world_frame(replay):
    final_frame = replay["frames"][-1]

    assert final_frame["state"][0] > 1.0
    assert final_frame["local_plan"][0][0] > 1.0
    assert final_frame["optimal_trajectory"][0][0] > 1.0


def assert_frame_arrays_are_finite(frame):
    for key in ("state", "executed_path", "command", "local_plan", "optimal_trajectory"):
        assert np.all(np.isfinite(np.asarray(frame[key], dtype=np.float32)))
    for rollout in frame.get("rollouts", []):
        assert np.all(np.isfinite(np.asarray(rollout, dtype=np.float32)))


def assert_replay_rollouts_are_finite_and_world_frame(replay):
    final_frame = replay["frames"][-1]

    assert final_frame["state"][0] > 1.0
    for rollout in final_frame["rollouts"]:
        assert rollout[0][0] > 1.0
        assert np.all(np.isfinite(np.asarray(rollout, dtype=np.float32)))


def recompute_minimum_clearance_from_result_volume(result):
    volume = BoxUnionVolume3D.from_config(result.robot_volume_config)
    clearances = []
    for state in result.state_history:
        body_points = transfer_from_global_to_local_frame(
            result.global_obstacle_points,
            state,
        )
        distances = volume.signed_distance(jnp.asarray(body_points, dtype=jnp.float32))
        clearances.append(float(jax.device_get(jnp.min(distances))))
    return min(clearances)


def assert_t_volume_has_nonconvex_notch(robot_volume_config):
    volume = BoxUnionVolume3D.from_config(robot_volume_config)
    probe_points = jnp.asarray(
        [
            [-0.08, 0.0, 0.0],
            [0.22, 0.25, 0.0],
            [-0.08, 0.25, 0.0],
        ],
        dtype=jnp.float32,
    )
    stem_inside, crossbar_inside, notch_outside = jax.device_get(
        volume.signed_distance(probe_points)
    )

    assert stem_inside < 0.0
    assert crossbar_inside < 0.0
    assert notch_outside > 0.0
