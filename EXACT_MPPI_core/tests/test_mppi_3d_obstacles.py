import numpy as np

from exact_mppi.mppi_3d import (
    BoxUnionVolume3D,
    ControlConstraints3D,
    MPPIController3D,
    ObstaclesCriticParams3D,
    OptimizerSettings3D,
    OptimalTrajectoryValidator3D,
    SamplingStd3D,
    Trajectories3D,
    ValidationResult3D,
    minimum_signed_distance_from_trajectories_to_obstacle_points,
    obstacles_critic_score_3d,
)


def _unit_box_volume():
    return BoxUnionVolume3D.from_config(
        [{"center": [0.0, 0.0, 0.0], "size": [1.0, 1.0, 1.0]}]
    )


def test_obstacle_cost_rises_for_colliding_trajectory_and_ignores_masked_points():
    volume = _unit_box_volume()
    trajectories = Trajectories3D(
        x=np.array([[0.0, 0.0], [0.55, 0.55], [2.0, 2.0]], dtype=np.float32),
        y=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        z=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        yaws=np.zeros((3, 2), dtype=np.float32),
    )
    obstacle_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    masked_distances = minimum_signed_distance_from_trajectories_to_obstacle_points(
        trajectories,
        obstacle_points,
        np.array([1.0, 0.0], dtype=np.float32),
        volume,
    )
    unmasked_distances = minimum_signed_distance_from_trajectories_to_obstacle_points(
        trajectories,
        obstacle_points,
        np.array([1.0, 1.0], dtype=np.float32),
        volume,
    )

    np.testing.assert_allclose(np.asarray(masked_distances[0]), [-0.5, -0.5])
    np.testing.assert_allclose(
        np.asarray(masked_distances[1]),
        [0.05, 0.05],
        atol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(masked_distances[2]), [1.5, 1.5])
    np.testing.assert_allclose(np.asarray(unmasked_distances[2]), [-0.5, -0.5])

    costs, _, fail_flag = obstacles_critic_score_3d(
        trajectories,
        obstacle_points,
        np.array([1.0, 0.0], dtype=np.float32),
        volume,
        ObstaclesCriticParams3D(
            enabled=True,
            power=1,
            repulsion_weight=0.0,
            critical_weight=10.0,
            collision_cost=1000.0,
            collision_margin_distance=0.1,
            repulsion_distance=1.0,
        ),
    )

    assert np.asarray(costs)[0] > np.asarray(costs)[1]
    assert np.asarray(costs)[1] > np.asarray(costs)[2]
    assert bool(np.asarray(fail_flag)) is False


def test_3d_trajectory_validator_reports_minimum_clearance_against_margin():
    settings = OptimizerSettings3D(
        constraints=ControlConstraints3D(
            vx_max=1.0,
            vx_min=-1.0,
            vy=1.0,
            vz=1.0,
            wz=1.0,
        ),
        sampling_std=SamplingStd3D(vx=0.1, vy=0.1, vz=0.1, wz=0.1),
        model_dt=0.1,
        temperature=0.3,
        batch_size=4,
        time_steps=4,
        iteration_count=1,
        shift_control_sequence=True,
        goal_weight=1.0,
        goal_yaw_weight=1.0,
        path_weight=1.0,
        control_weight=1.0,
    )
    validator = OptimalTrajectoryValidator3D(
        settings,
        collision_lookahead_time=0.4,
        collision_margin_distance=0.2,
    )
    optimal_trajectory = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0, 0.0],
            [0.6, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    obstacle_points = np.array([[1.2, 0.0, 0.0]], dtype=np.float32)

    result, min_clearance = validator.validateTrajectory(
        optimal_trajectory,
        obstacle_points,
        np.array([1.0], dtype=np.float32),
        _unit_box_volume(),
    )

    assert int(np.asarray(result)) == int(ValidationResult3D.SOFT_RESET)
    np.testing.assert_allclose(np.asarray(min_clearance), 0.0999999, atol=1e-6)


def test_controller_reports_selected_trajectory_clearance():
    controller = MPPIController3D(
        model_dt=0.1,
        time_steps=8,
        batch_size=64,
        iteration_count=1,
        seed=11,
        robot_volume_config=[
            {"center": [0.0, 0.0, 0.0], "size": [0.4, 0.4, 0.4]},
        ],
        obstacles_critical_weight=5.0,
        obstacles_collision_margin_distance=0.05,
        TrajectoryValidator={
            "collision_lookahead_time": 0.8,
            "collision_margin_distance": 0.05,
        },
    )

    command = controller.computeVelocityCommands(
        robot_pose=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        robot_speed=np.zeros(4, dtype=np.float32),
        plan=np.column_stack(
            [
                np.linspace(0.0, 1.0, 8),
                np.zeros(8),
                np.zeros(8),
                np.zeros(8),
            ]
        ).astype(np.float32),
        goal=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        obstacle_points=np.array([[4.0, 0.0, 0.0]], dtype=np.float32),
    )

    assert command.shape == (4,)
    assert np.isfinite(command).all()
    assert controller.getLastMinimumClearance() is not None
    assert controller.getLastMinimumClearance() > 0.05
    assert controller.getLastTrajectoryValidationResult() == int(
        ValidationResult3D.SUCCESS
    )
