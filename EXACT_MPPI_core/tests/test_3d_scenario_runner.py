import json

import numpy as np

from exact_mppi.scenario_runner_3d import (
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
