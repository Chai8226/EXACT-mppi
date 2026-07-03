import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import exact_mppi.scenario_runner_3d as scenario_runner_3d
from exact_mppi.mppi_3d import BoxUnionVolume3D
from exact_mppi.scenario_runner_3d import (
    STATIC_3D_SCENARIOS,
    ScenarioRunResult3D,
    build_3d_baseline_report,
    build_3d_replay_data,
    build_mid360_like_observed_point_cloud,
    compute_3d_smoothness_telemetry,
    load_builtin_scenario_config,
    run_3d_scenario,
    run_3d_static_scenario_suite,
    select_global_plan,
    write_3d_baseline_replay_artifacts,
    write_3d_replay_json,
)


def test_mid360_like_box_raycast_returns_empty_cloud_for_empty_scene():
    observed = build_mid360_like_observed_point_cloud(
        [],
        np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        {
            "horizontal_samples": 4,
            "vertical_samples": 3,
        },
    )

    assert observed.shape == (0, 3)


def test_mid360_like_box_raycast_respects_range_and_vertical_fov():
    pose = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    sensor = {
        "min_range_m": 0.1,
        "max_range_m": 1.0,
        "horizontal_samples": 1,
        "vertical_samples": 1,
        "vertical_min_deg": 0.0,
        "vertical_max_deg": 0.0,
    }

    in_range = build_mid360_like_observed_point_cloud(
        [{"type": "box", "center": [0.75, 0.0, 0.0], "size": [0.2, 0.2, 0.2]}],
        pose,
        sensor,
    )
    too_far = build_mid360_like_observed_point_cloud(
        [{"type": "box", "center": [1.3, 0.0, 0.0], "size": [0.2, 0.2, 0.2]}],
        pose,
        sensor,
    )
    too_near = build_mid360_like_observed_point_cloud(
        [{"type": "box", "center": [0.05, 0.0, 0.0], "size": [0.04, 0.04, 0.04]}],
        pose,
        sensor,
    )
    above_fov = build_mid360_like_observed_point_cloud(
        [{"type": "box", "center": [0.75, 0.0, 0.5], "size": [0.2, 0.2, 0.2]}],
        pose,
        sensor,
    )
    side_out_of_fov = build_mid360_like_observed_point_cloud(
        [{"type": "box", "center": [0.0, 0.75, 0.0], "size": [0.2, 0.2, 0.2]}],
        pose,
        {
            **sensor,
            "horizontal_fov_deg": 30.0,
        },
    )

    assert in_range.shape == (1, 3)
    np.testing.assert_allclose(in_range[0], [0.65, 0.0, 0.0], atol=1e-6)
    assert too_far.shape == (0, 3)
    assert too_near.shape == (0, 3)
    assert above_fov.shape == (0, 3)
    assert side_out_of_fov.shape == (0, 3)


def test_mid360_like_box_raycast_respects_default_asymmetric_vertical_fov():
    def point_at_elevation(elevation_deg):
        elevation = np.deg2rad(elevation_deg)
        return [float(np.cos(elevation)), 0.0, float(np.sin(elevation))]

    sensor = {
        "horizontal_samples": 1,
        "vertical_samples": 2,
    }
    observed = build_mid360_like_observed_point_cloud(
        [
            {
                "type": "box",
                "center": point_at_elevation(-7.0),
                "size": [0.08, 0.08, 0.08],
            },
            {
                "type": "box",
                "center": point_at_elevation(52.0),
                "size": [0.08, 0.08, 0.08],
            },
        ],
        np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        sensor,
    )
    below_fov = build_mid360_like_observed_point_cloud(
        [
            {
                "type": "box",
                "center": point_at_elevation(-12.0),
                "size": [0.04, 0.04, 0.04],
            }
        ],
        np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        sensor,
    )
    above_fov = build_mid360_like_observed_point_cloud(
        [
            {
                "type": "box",
                "center": point_at_elevation(57.0),
                "size": [0.04, 0.04, 0.04],
            }
        ],
        np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        sensor,
    )

    assert observed.shape == (2, 3)
    assert below_fov.shape == (0, 3)
    assert above_fov.shape == (0, 3)


def test_mid360_like_box_raycast_respects_default_range_limits():
    sensor = {
        "horizontal_samples": 1,
        "vertical_samples": 1,
        "vertical_min_deg": 0.0,
        "vertical_max_deg": 0.0,
    }

    in_range = build_mid360_like_observed_point_cloud(
        [{"type": "box", "center": [5.0, 0.0, 0.0], "size": [0.2, 0.2, 0.2]}],
        np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        sensor,
    )
    too_near = build_mid360_like_observed_point_cloud(
        [{"type": "box", "center": [0.05, 0.0, 0.0], "size": [0.04, 0.04, 0.04]}],
        np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        sensor,
    )
    too_far = build_mid360_like_observed_point_cloud(
        [{"type": "box", "center": [10.3, 0.0, 0.0], "size": [0.2, 0.2, 0.2]}],
        np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        sensor,
    )

    assert in_range.shape == (1, 3)
    np.testing.assert_allclose(in_range[0], [4.9, 0.0, 0.0], atol=1e-6)
    assert too_near.shape == (0, 3)
    assert too_far.shape == (0, 3)


def test_mid360_like_zero_noise_dropout_and_seed_are_deterministic():
    geometry = [
        {"type": "box", "center": [1.0, 0.0, 0.0], "size": [0.2, 0.2, 0.2]},
    ]
    sensor = {
        "horizontal_samples": 1,
        "vertical_samples": 1,
        "vertical_min_deg": 0.0,
        "vertical_max_deg": 0.0,
        "noise_std_m": 0.0,
        "dropout_probability": 0.0,
        "seed": 123,
    }

    default_observed = build_mid360_like_observed_point_cloud(
        geometry,
        np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        {
            "horizontal_samples": 1,
            "vertical_samples": 1,
            "vertical_min_deg": 0.0,
            "vertical_max_deg": 0.0,
        },
    )
    first = build_mid360_like_observed_point_cloud(
        geometry,
        np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        sensor,
    )
    second = build_mid360_like_observed_point_cloud(
        geometry,
        np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        sensor,
    )

    np.testing.assert_allclose(first, default_observed)
    np.testing.assert_allclose(second, first)


def test_mid360_like_nonzero_noise_and_dropout_are_rejected():
    base_sensor = {
        "horizontal_samples": 1,
        "vertical_samples": 1,
        "vertical_min_deg": 0.0,
        "vertical_max_deg": 0.0,
    }

    with pytest.raises(ValueError, match="noise is reserved"):
        build_mid360_like_observed_point_cloud(
            [],
            np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            {
                **base_sensor,
                "noise_std_m": 0.01,
            },
        )
    with pytest.raises(ValueError, match="dropout is reserved"):
        build_mid360_like_observed_point_cloud(
            [],
            np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            {
                **base_sensor,
                "dropout_probability": 0.1,
            },
        )


def test_mid360_like_box_raycast_keeps_nearest_occluding_hit_per_ray():
    observed = build_mid360_like_observed_point_cloud(
        [
            {"type": "box", "center": [0.7, 0.0, 0.0], "size": [0.2, 0.2, 0.2]},
            {"type": "box", "center": [1.2, 0.0, 0.0], "size": [0.2, 0.2, 0.2]},
        ],
        np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        {
            "horizontal_samples": 1,
            "vertical_samples": 1,
            "vertical_min_deg": 0.0,
            "vertical_max_deg": 0.0,
        },
    )

    assert observed.shape == (1, 3)
    np.testing.assert_allclose(observed[0], [0.6, 0.0, 0.0], atol=1e-6)


def test_mid360_like_runner_feeds_robot_local_yaw_frame_observations(monkeypatch):
    captured_obstacle_points = []

    class FakeController:
        def computeVelocityCommands(self, **kwargs):
            captured_obstacle_points.append(
                np.asarray(kwargs["obstacle_points"], dtype=np.float32)
            )
            return np.zeros(4, dtype=np.float32)

        def getOptimalTrajectory(self):
            return None

    monkeypatch.setattr(
        scenario_runner_3d,
        "_build_controller",
        lambda *_, **__: FakeController(),
    )
    config = make_mid360_tracer_bullet_config()

    result = run_3d_scenario(config)
    replay = build_3d_replay_data(result)

    assert result.step_count == 1
    assert len(captured_obstacle_points) == 1
    assert captured_obstacle_points[0].shape == (1, 3)
    np.testing.assert_allclose(
        captured_obstacle_points[0][0],
        [0.8, 0.0, 0.0],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result.observed_point_cloud_history[0],
        [[1.0, 2.8, 0.0]],
    )
    assert replay["scene"]["obstacle_geometry"] == config["obstacles"]["geometry"]
    assert "obstacle_points" not in replay["scene"]
    np.testing.assert_allclose(
        replay["frames"][0]["observed_point_cloud"],
        [[1.0, 2.8, 0.0]],
    )
    json.dumps(replay, allow_nan=False)


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
    config = load_builtin_scenario_config("narrow_gap_t_volume_3d")
    result = run_3d_scenario(config)

    replay = build_3d_replay_data(result)
    replay_path = tmp_path / "narrow_gap_t_volume_3d.replay.json"
    write_3d_replay_json(result, replay_path)

    assert json.loads(replay_path.read_text(encoding="utf-8")) == replay
    assert_replay_scene(replay, result)
    assert_replay_frames(replay, result, require_clearance=True)
    assert_replay_geometry_is_world_frame(replay)
    assert replay["scene"]["robot_volume"]["boxes"] == result.robot_volume_config
    assert len(replay["frames"][0]["reference_window"]) == config_time_steps(config)
    first_reference_window = np.asarray(
        replay["frames"][0]["reference_window"],
        dtype=np.float32,
    )
    assert first_reference_window[-1, 0] - first_reference_window[0, 0] >= 1.0


def test_narrow_gap_t_volume_3d_exports_obstacle_geometry_separate_from_points():
    result = run_3d_scenario(load_builtin_scenario_config("narrow_gap_t_volume_3d"))

    replay = build_3d_replay_data(result)
    scene = replay["scene"]

    assert "obstacle_points" not in scene
    assert scene["obstacle_geometry"] == result.obstacle_geometry_config
    assert len(scene["obstacle_geometry"]) == 2
    for obstacle in scene["obstacle_geometry"]:
        assert obstacle["type"] == "box"
        assert len(obstacle["center"]) == 3
        assert len(obstacle["size"]) == 3
        assert all(np.isfinite(value) for value in obstacle["center"])
        assert all(value > 0 for value in obstacle["size"])


def test_dense_reference_path_window_spans_configured_physical_lookahead():
    reference_path = np.asarray(
        [[x / 10.0, 0.0, 0.0, 0.0] for x in range(31)],
        dtype=np.float32,
    )

    window = select_global_plan(
        reference_path,
        np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        time_steps=5,
        reference_window_lookahead_distance=1.0,
    )

    assert window.shape == (5, 4)
    np.testing.assert_allclose(window[:, 0], [0.0, 0.25, 0.5, 0.75, 1.0])


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

    report = build_3d_baseline_report(results)

    assert report["kind"] == "3d_static_scenario_baseline"
    assert report["aggregate"]["scenario_count"] == len(STATIC_3D_SCENARIOS)
    assert report["aggregate"]["pass_count"] + report["aggregate"]["fail_count"] == len(
        STATIC_3D_SCENARIOS
    )
    assert report["aggregate"]["failed_scenarios"] == [
        scenario["scenario"]
        for scenario in report["scenarios"]
        if scenario["status"] == "fail"
    ]
    assert "needs_followup_tuning_scenarios" in report["aggregate"]
    for scenario in report["scenarios"]:
        assert_standard_summary(scenario, scenario["scenario"])
        assert scenario["status"] in ("pass", "fail")
        assert isinstance(scenario["needs_followup_tuning"], bool)
        assert isinstance(scenario["followup_tuning_reasons"], list)


def test_3d_baseline_report_identifies_followup_tuning_reasons():
    passing = make_minimal_result(
        "open_track_3d",
        reached_goal=True,
        collided=False,
        final_distance=0.1,
        command_delta=0.2,
        trajectory_bend=0.05,
    )
    missed = make_minimal_result(
        "t_shape_trap_3d",
        reached_goal=False,
        collided=False,
        final_distance=2.0,
        command_delta=1.0,
        trajectory_bend=0.4,
    )
    collided = make_minimal_result(
        "cluttered_corridor_3d",
        reached_goal=True,
        collided=True,
        final_distance=0.2,
        command_delta=0.2,
        trajectory_bend=0.05,
    )

    report = build_3d_baseline_report([passing, missed, collided])

    assert report["aggregate"]["pass_count"] == 1
    assert report["aggregate"]["fail_count"] == 2
    assert report["aggregate"]["missed_goal_scenarios"] == ["t_shape_trap_3d"]
    assert report["aggregate"]["collided_scenarios"] == ["cluttered_corridor_3d"]
    assert "t_shape_trap_3d" in report["aggregate"]["poor_smoothness_scenarios"]
    assert report["aggregate"]["needs_followup_tuning_scenarios"] == [
        "t_shape_trap_3d",
        "cluttered_corridor_3d",
    ]
    trap = report["scenarios"][1]
    assert trap["status"] == "fail"
    assert trap["followup_tuning_reasons"] == [
        "missed_goal",
        "poor_command_smoothness",
        "poor_trajectory_smoothness",
    ]


def test_3d_baseline_replay_artifacts_can_be_written(tmp_path):
    result = run_3d_scenario(load_builtin_scenario_config("open_track_3d"))

    artifacts = write_3d_baseline_replay_artifacts([result], tmp_path)

    assert artifacts == {
        "open_track_3d": str(tmp_path / "open_track_3d.replay.json")
    }
    replay = json.loads(
        (tmp_path / "open_track_3d.replay.json").read_text(encoding="utf-8")
    )
    assert_replay_scene(replay, result)
    assert_replay_frames(replay, result, require_clearance=False)


def test_all_static_cli_writes_baseline_report_and_replay_artifacts(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        scenario_runner_3d,
        "run_3d_static_scenario_suite",
        lambda **_: [
            make_minimal_result(
                "open_track_3d",
                reached_goal=True,
                collided=False,
                final_distance=0.1,
            ),
            make_minimal_result(
                "t_shape_trap_3d",
                reached_goal=False,
                collided=False,
                final_distance=2.0,
            ),
        ],
    )
    report_path = tmp_path / "baseline.json"
    replay_dir = tmp_path / "replays"

    exit_code = scenario_runner_3d.main(
        [
            "--all-static",
            "--summary-json",
            str(report_path),
            "--replay-dir",
            str(replay_dir),
        ]
    )

    assert exit_code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["aggregate"]["scenario_count"] == 2
    assert report["aggregate"]["failed_scenarios"] == ["t_shape_trap_3d"]
    assert report["scenarios"][0]["replay_json"] == str(
        replay_dir / "open_track_3d.replay.json"
    )
    assert (replay_dir / "open_track_3d.replay.json").is_file()
    assert (replay_dir / "t_shape_trap_3d.replay.json").is_file()


def test_static_3d_scenario_suite_configs_do_not_define_dynamic_obstacles():
    for scenario_name in STATIC_3D_SCENARIOS:
        config = load_builtin_scenario_config(scenario_name)

        assert "dynamic_obstacles" not in config
        assert "dynamic_obstacles" not in config.get("obstacles", {})


def test_static_3d_scenario_descriptions_do_not_use_legacy_observation_language():
    legacy_phrases = (
        "range-based",
        "static obstacle points",
        "global obstacle points",
        "local obstacle oracle",
    )
    for scenario_name in STATIC_3D_SCENARIOS:
        config = load_builtin_scenario_config(scenario_name)
        description = str(config.get("description", "")).lower()

        for phrase in legacy_phrases:
            assert phrase not in description


def test_static_3d_scenario_suite_configs_use_mid360_sensor_config():
    for scenario_name in STATIC_3D_SCENARIOS:
        config = load_builtin_scenario_config(scenario_name)

        assert "observation_range" not in config.get("simulation", {})
        assert "max_obstacle_points" not in config.get("simulation", {})
        expected_sensor = {
            "type": "mid360_like",
            "min_range_m": 0.1,
            "max_range_m": 10.0,
            "horizontal_fov_deg": 360.0,
            "vertical_min_deg": -7.0,
            "vertical_max_deg": 52.0,
            "horizontal_samples": (
                48 if scenario_name == "narrow_gap_t_volume_3d" else 36
            ),
            "vertical_samples": 8,
            "noise_std_m": 0.0,
            "dropout_probability": 0.0,
        }
        assert config["sensor"] == expected_sensor
        assert int(config["controller"]["max_obs_num"]) > 0


def test_migrated_builtin_config_does_not_need_legacy_points_for_perception(
    monkeypatch,
):
    captured_obstacle_points = []

    class FakeController:
        def computeVelocityCommands(self, **kwargs):
            captured_obstacle_points.append(
                np.asarray(kwargs["obstacle_points"], dtype=np.float32)
            )
            return np.zeros(4, dtype=np.float32)

        def getOptimalTrajectory(self):
            return None

    monkeypatch.setattr(
        scenario_runner_3d,
        "_build_controller",
        lambda *_, **__: FakeController(),
    )
    config = load_builtin_scenario_config("narrow_gap_t_volume_3d")
    config["simulation"]["max_steps"] = 1
    config["obstacles"].pop("points", None)

    result = run_3d_scenario(config)

    assert result.global_obstacle_points.shape == (0, 3)
    assert result.observed_point_cloud_history[0].shape[0] > 0
    assert captured_obstacle_points[0].shape[0] > 0


def test_geometry_truth_reports_collision_when_observed_cloud_is_empty(monkeypatch):
    class FakeController:
        def computeVelocityCommands(self, **_):
            return np.zeros(4, dtype=np.float32)

        def getOptimalTrajectory(self):
            return None

    monkeypatch.setattr(
        scenario_runner_3d,
        "_build_controller",
        lambda *_, **__: FakeController(),
    )

    config = make_mid360_tracer_bullet_config()
    config["simulation"]["max_steps"] = 1
    config["simulation"]["clearance_margin"] = 0.04
    config["sensor"]["min_range_m"] = 1.0
    config["obstacles"] = {
        "geometry": [
            {
                "type": "box",
                "center": [1.0, 2.0, 0.0],
                "size": [0.2, 0.2, 0.2],
            }
        ],
        "points": [],
    }

    result = run_3d_scenario(config)
    replay = build_3d_replay_data(result)
    report = build_3d_baseline_report([result])

    assert result.global_obstacle_points.shape == (0, 3)
    assert result.observed_point_cloud_history[0].shape == (0, 3)
    assert result.minimum_clearance is not None
    assert result.minimum_clearance < 0.0
    assert result.collided is True
    assert replay["summary"]["collided"] is True
    assert np.isfinite(replay["frames"][0]["clearance"])
    assert report["aggregate"]["collision_count"] == 1
    assert report["aggregate"]["collided_scenarios"] == ["mid360_tracer_bullet"]


def test_mid360_config_requires_controller_point_budget():
    config = load_builtin_scenario_config("narrow_gap_t_volume_3d")
    del config["controller"]["max_obs_num"]

    with pytest.raises(ValueError, match="controller.max_obs_num"):
        run_3d_scenario(config)


def test_new_3d_observation_path_requires_sensor_config():
    config = load_builtin_scenario_config("narrow_gap_t_volume_3d")
    del config["sensor"]

    with pytest.raises(ValueError, match="top-level sensor"):
        run_3d_scenario(config)


def test_new_3d_observation_path_rejects_legacy_observation_fields():
    config = load_builtin_scenario_config("narrow_gap_t_volume_3d")
    del config["sensor"]
    config["simulation"]["observation_range"] = 1.8
    config["simulation"]["max_obstacle_points"] = 64

    with pytest.raises(ValueError, match="top-level sensor"):
        run_3d_scenario(config)


def test_mid360_config_rejects_unsupported_sensor_type():
    config = load_builtin_scenario_config("narrow_gap_t_volume_3d")
    config["sensor"]["type"] = "depth_camera"

    with pytest.raises(ValueError, match="Unsupported 3D sensor type"):
        run_3d_scenario(config)


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
    assert "obstacle_points" not in scene
    assert scene["obstacle_geometry"] == result.obstacle_geometry_config
    assert scene["robot_volume"]["type"] == "box_union"
    assert scene["robot_volume"]["boxes"] == result.robot_volume_config
    assert "sensor" not in scene
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
        assert frame["reference_window"] == result.local_plan_history[frame_index].tolist()
        assert frame["local_plan"] == result.local_plan_history[frame_index].tolist()
        assert (
            frame["observed_point_cloud"]
            == result.observed_point_cloud_history[frame_index].tolist()
        )
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
    for key in (
        "state",
        "executed_path",
        "command",
        "reference_window",
        "local_plan",
        "optimal_trajectory",
        "observed_point_cloud",
    ):
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
        clearances.append(
            scenario_runner_3d._minimum_state_clearance(
                volume,
                result.robot_volume_config,
                result.obstacle_geometry_config,
                result.global_obstacle_points,
                state,
            )
        )
    return min(clearances)


def config_time_steps(config):
    return int(config.get("controller", {}).get("time_steps", 12))


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


def make_minimal_result(
    scenario,
    *,
    reached_goal,
    collided,
    final_distance,
    command_delta=0.1,
    trajectory_bend=0.0,
):
    command_history = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [command_delta, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    state_history = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [1.0 + trajectory_bend, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    local_plan_history = np.asarray(
        [
            [[0.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0]],
            [[0.5, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    return ScenarioRunResult3D(
        scenario=scenario,
        reached_goal=reached_goal,
        collided=collided,
        final_distance=final_distance,
        minimum_clearance=None,
        step_count=2,
        final_state=state_history[-1],
        state_history=state_history,
        command_history=command_history,
        local_plan_history=local_plan_history,
        optimal_trajectory_history=local_plan_history,
        rollout_history=[],
        clearance_history=[None, None, None],
        global_reference_path=np.asarray(
            [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        global_obstacle_points=np.empty((0, 3), dtype=np.float32),
        observed_point_cloud_history=[
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
        ],
        obstacle_geometry_config=[],
        robot_volume_config=[
            {
                "center": [0.0, 0.0, 0.0],
                "size": [0.4, 0.3, 0.2],
            }
        ],
    )


def make_mid360_tracer_bullet_config():
    return {
        "name": "mid360_tracer_bullet",
        "simulation": {
            "model_dt": 0.1,
            "max_steps": 1,
            "goal_tolerance": 0.01,
            "clearance_margin": 0.04,
        },
        "sensor": {
            "type": "mid360_like",
            "min_range_m": 0.1,
            "max_range_m": 10.0,
            "horizontal_fov_deg": 360.0,
            "vertical_min_deg": 0.0,
            "vertical_max_deg": 0.0,
            "horizontal_samples": 1,
            "vertical_samples": 1,
        },
        "robot_volume": {
            "boxes": [
                {
                    "center": [0.0, 0.0, 0.0],
                    "size": [0.35, 0.35, 0.35],
                }
            ]
        },
        "reference_path": {
            "point_count": 2,
            "waypoints": [
                [1.0, 2.0, 0.0, np.pi / 2.0],
                [1.0, 2.1, 0.0, np.pi / 2.0],
            ],
        },
        "obstacles": {
            "geometry": [
                {
                    "type": "box",
                    "center": [1.0, 2.9, 0.0],
                    "size": [0.2, 0.2, 0.2],
                }
            ],
            "points": [
                [99.0, 0.0, 0.0],
            ],
        },
        "controller": {
            "time_steps": 2,
            "max_obs_num": 4,
        },
    }
