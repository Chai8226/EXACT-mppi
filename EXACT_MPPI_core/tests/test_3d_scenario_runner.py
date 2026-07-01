import json

import numpy as np

from exact_mppi.scenario_runner_3d import (
    compute_3d_smoothness_telemetry,
    load_builtin_scenario_config,
    run_3d_scenario,
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
