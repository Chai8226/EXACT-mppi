import numpy as np

from exact_mppi.mppi_3d import (
    PreferForwardCriticParams3D,
    TwirlingCriticParams3D,
    VelocityDeadbandCriticParams3D,
    prefer_forward_critic_score_3d,
    twirling_critic_score_3d,
    velocity_deadband_critic_score_3d,
)


def test_prefer_forward_critic_penalizes_reverse_body_x_velocity_only_when_path_is_long():
    params = PreferForwardCriticParams3D(
        enabled=True,
        power=1,
        weight=2.0,
        threshold_to_consider=0.5,
    )

    costs, _ = prefer_forward_critic_score_3d(
        vx=np.array([[0.3, 0.2], [-0.4, -0.1]], dtype=np.float32),
        local_path_length=np.array(1.0, dtype=np.float32),
        params=params,
        model_dt=0.5,
    )
    short_path_costs, _ = prefer_forward_critic_score_3d(
        vx=np.array([[-0.4, -0.1]], dtype=np.float32),
        local_path_length=np.array(0.1, dtype=np.float32),
        params=params,
        model_dt=0.5,
    )

    np.testing.assert_allclose(np.asarray(costs)[0], 0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(costs)[1], 0.5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(short_path_costs), [0.0], atol=1e-6)


def test_velocity_deadband_critic_scores_all_4d_yaw_only_3d_controls():
    params = VelocityDeadbandCriticParams3D(
        enabled=True,
        power=1,
        weight=10.0,
        deadband_velocities=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
    )

    costs, _ = velocity_deadband_critic_score_3d(
        vx=np.array([[0.2, -0.2], [0.05, -0.05]], dtype=np.float32),
        vy=np.array([[0.3, -0.3], [0.10, -0.10]], dtype=np.float32),
        vz=np.array([[0.4, -0.4], [0.15, -0.15]], dtype=np.float32),
        wz=np.array([[0.5, -0.5], [0.20, -0.20]], dtype=np.float32),
        params=params,
        model_dt=0.5,
    )

    np.testing.assert_allclose(np.asarray(costs)[0], 0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(costs)[1], 5.0, atol=1e-6)


def test_twirling_critic_penalizes_unnecessary_yaw_rate_when_path_is_long():
    params = TwirlingCriticParams3D(
        enabled=True,
        power=1,
        weight=3.0,
        pose_tolerance=0.2,
    )

    costs, _ = twirling_critic_score_3d(
        wz=np.array([[0.0, 0.0], [0.2, -0.4]], dtype=np.float32),
        local_path_length=np.array(1.0, dtype=np.float32),
        params=params,
    )
    short_path_costs, _ = twirling_critic_score_3d(
        wz=np.array([[0.2, -0.4]], dtype=np.float32),
        local_path_length=np.array(0.1, dtype=np.float32),
        params=params,
    )

    np.testing.assert_allclose(np.asarray(costs)[0], 0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(costs)[1], 0.9, atol=1e-6)
    np.testing.assert_allclose(np.asarray(short_path_costs), [0.0], atol=1e-6)
