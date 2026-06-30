import numpy as np

import exact_mppi.mppi_3d as mppi_3d
from exact_mppi.mppi_3d import (
    ConstraintCriticParams3D,
    ControlConstraints3D,
    GoalCriticParams3D,
    GoalYawCriticParams3D,
    PathAlignCriticParams3D,
    PathFollowCriticParams3D,
    Trajectories3D,
    constraint_critic_score_3d,
    goal_critic_score_3d,
    goal_yaw_critic_score_3d,
    path_align_critic_score_3d,
    path_follow_critic_score_3d,
)


def test_constraint_critic_scores_4d_yaw_only_3d_control_violations():
    constraints = ControlConstraints3D(
        vx_max=1.0,
        vx_min=-0.5,
        vy=0.4,
        vz=0.3,
        wz=0.2,
    )
    params = ConstraintCriticParams3D(enabled=True, power=1, weight=2.0)

    costs, _ = constraint_critic_score_3d(
        vx=np.array([[0.5, -0.2], [1.2, -0.8]], dtype=np.float32),
        vy=np.array([[0.1, -0.1], [0.5, 0.0]], dtype=np.float32),
        vz=np.array([[0.2, -0.2], [0.0, -0.5]], dtype=np.float32),
        wz=np.array([[0.1, -0.1], [0.3, 0.0]], dtype=np.float32),
        constraints=constraints,
        params=params,
        model_dt=0.5,
    )

    np.testing.assert_allclose(np.asarray(costs)[0], 0.0, atol=1e-6)
    assert np.asarray(costs)[1] > 0.0


def test_goal_critic_uses_3d_position_distance_and_ignores_yaw():
    trajectories = Trajectories3D(
        x=np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
        y=np.zeros((2, 2), dtype=np.float32),
        z=np.array([[1.0, 1.0], [3.0, 3.0]], dtype=np.float32),
        yaws=np.array([[0.0, 2.0], [0.0, -2.0]], dtype=np.float32),
    )

    costs, _ = goal_critic_score_3d(
        trajectories,
        goal=np.array([1.0, 0.0, 1.0, -3.0], dtype=np.float32),
        params=GoalCriticParams3D(enabled=True, power=1, weight=1.0),
    )

    np.testing.assert_allclose(np.asarray(costs)[0], 0.0, atol=1e-6)
    assert np.asarray(costs)[1] > 1.9


def test_goal_yaw_critic_is_yaw_only_and_active_only_near_goal_position():
    trajectories = Trajectories3D(
        x=np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        y=np.zeros((2, 2), dtype=np.float32),
        z=np.array([[0.0, 5.0], [0.0, -5.0]], dtype=np.float32),
        yaws=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
    )
    params = GoalYawCriticParams3D(
        enabled=True,
        power=1,
        weight=1.0,
        threshold_to_consider=0.5,
    )
    goal = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    far_costs, _ = goal_yaw_critic_score_3d(
        trajectories,
        current_pose=np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32),
        goal=goal,
        params=params,
    )
    near_costs, _ = goal_yaw_critic_score_3d(
        trajectories,
        current_pose=np.array([0.1, 0.0, 0.0, 0.0], dtype=np.float32),
        goal=goal,
        params=params,
    )

    np.testing.assert_allclose(np.asarray(far_costs), np.zeros(2), atol=1e-6)
    np.testing.assert_allclose(np.asarray(near_costs)[0], 0.0, atol=1e-6)
    assert np.asarray(near_costs)[1] > 0.9


def test_path_align_critic_is_position_only_in_3d():
    plan = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 2.0],
        ],
        dtype=np.float32,
    )
    trajectories = Trajectories3D(
        x=np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float32),
        y=np.zeros((3, 2), dtype=np.float32),
        z=np.array([[0.0, 1.0], [0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
        yaws=np.array([[0.0, 0.0], [3.0, -3.0], [0.0, 0.0]], dtype=np.float32),
    )

    costs, _ = path_align_critic_score_3d(
        trajectories,
        plan,
        PathAlignCriticParams3D(
            enabled=True,
            power=1,
            weight=1.0,
            trajectory_point_step=1,
        ),
    )

    np.testing.assert_allclose(np.asarray(costs)[0], np.asarray(costs)[1], atol=1e-6)
    assert np.asarray(costs)[2] > np.asarray(costs)[0]


def test_path_follow_critic_measures_progress_against_3d_reference_path():
    plan = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 1.0],
            [2.0, 0.0, 2.0, -1.0],
        ],
        dtype=np.float32,
    )
    trajectories = Trajectories3D(
        x=np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float32),
        y=np.zeros((2, 2), dtype=np.float32),
        z=np.array([[1.0, 2.0], [1.0, 0.0]], dtype=np.float32),
        yaws=np.array([[0.0, 0.0], [2.0, -2.0]], dtype=np.float32),
    )

    costs, info = path_follow_critic_score_3d(
        trajectories,
        current_pose=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        plan=plan,
        params=PathFollowCriticParams3D(
            enabled=True,
            power=1,
            weight=1.0,
            offset_from_nearest=2,
        ),
    )

    np.testing.assert_allclose(np.asarray(info["path_follow_point"]), [2.0, 0.0, 2.0])
    np.testing.assert_allclose(np.asarray(costs)[0], 0.0, atol=1e-6)
    assert np.asarray(costs)[1] > 1.9


def test_3d_critic_set_does_not_include_path_angle_critic():
    assert not hasattr(mppi_3d, "PathAngleCritic3D")
