import importlib.util
from pathlib import Path

import numpy as np


def _load_example_module():
    example_path = (
        Path(__file__).resolve().parents[1]
        / "example"
        / "yaw_only_3d_obstacle_avoidance"
        / "mppi_3d_obstacle_avoidance.py"
    )
    spec = importlib.util.spec_from_file_location(
        "mppi_3d_obstacle_avoidance",
        example_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_core_only_3d_obstacle_avoidance_example_reaches_goal_headlessly():
    example = _load_example_module()

    result = example.run_3d_obstacle_avoidance_example(max_steps=80)

    assert result.reached_goal
    assert not result.collided
    assert result.min_sdf_clearance >= 0.04
    assert result.global_obstacle_points.shape[1] == 3
    assert result.global_reference_path.shape[1] == 4
    assert result.command_history.shape[1] == 4
    assert result.visualization_frames == ()
    assert (
        np.linalg.norm(result.final_state[:3] - result.global_reference_path[-1, :3])
        <= 0.28
    )


def test_3d_example_cli_defaults_to_render_and_accepts_2d_style_aliases():
    example = _load_example_module()
    parser = example.build_arg_parser()

    defaults = parser.parse_args([])
    aliased = parser.parse_args(["--no-render", "-a", "--show_rollouts"])

    assert defaults.render is True
    assert defaults.save_gif is False
    assert defaults.show_rollouts is False
    assert aliased.render is False
    assert aliased.save_gif is True
    assert aliased.show_rollouts is True


def test_3d_example_can_save_visualization_gif_headlessly(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    example = _load_example_module()
    gif_path = tmp_path / "yaw_only_3d_mppi.gif"

    result = example.run_3d_obstacle_avoidance_example(
        max_steps=2,
        render=False,
        save_gif=True,
        gif_path=gif_path,
        show_rollouts=True,
    )

    assert result.state_history.shape[0] >= 2
    assert gif_path.exists()
    assert gif_path.stat().st_size > 0


def test_3d_example_builds_range_based_local_observation_with_mask():
    example = _load_example_module()
    obstacle_points = example.build_global_obstacle_points()

    local_points, mask = example.build_range_based_local_observation(
        obstacle_points,
        robot_pose=np.array([2.4, 0.0, 0.3, 0.0], dtype=np.float32),
        observation_range=0.45,
        max_points=24,
    )

    assert local_points.shape == (24, 3)
    assert mask.shape == (24,)
    assert 0 < int(mask.sum()) <= 24
    assert np.all(np.isfinite(local_points))
