import math

import jax.numpy as jnp
import numpy as np

import exact_mppi.mppi_3d.controller as mppi_3d_controller
from exact_mppi.mppi_jax.controller import MPPIController
from exact_mppi.mppi_3d import MPPIController3D, YawOnly3DHolonomicMotionModel
from exact_mppi.mppi_3d.models import ControlSequence3D


def test_existing_2d_controller_import_still_works():
    assert MPPIController is not None


def test_yaw_only_3d_motion_model_integrates_body_frame_controls():
    model = YawOnly3DHolonomicMotionModel(model_dt=0.5)
    controls = ControlSequence3D(
        vx=jnp.array([2.0, 0.0]),
        vy=jnp.array([0.0, 1.0]),
        vz=jnp.array([0.5, -0.5]),
        wz=jnp.array([math.pi, 0.0]),
    )

    trajectory = model.integrate_sequence(
        controls,
        jnp.array([1.0, 2.0, 3.0, math.pi / 2.0], dtype=jnp.float32),
    )

    np.testing.assert_allclose(
        np.asarray(trajectory),
        np.array(
            [
                [1.0, 3.0, 3.25, math.pi],
                [1.0, 2.5, 3.0, math.pi],
            ],
            dtype=np.float32,
        ),
        rtol=1e-6,
        atol=1e-6,
    )


def test_3d_controller_returns_finite_4d_command_in_no_obstacle_loop():
    controller = MPPIController3D(
        model_dt=0.1,
        time_steps=12,
        batch_size=128,
        iteration_count=2,
        seed=7,
        vx_std=0.35,
        vy_std=0.25,
        vz_std=0.25,
        wz_std=0.2,
        vx_max=1.0,
        vx_min=-0.4,
        vy_max=0.8,
        vz_max=0.8,
        wz_max=1.0,
        goal_weight=10.0,
        path_weight=1.0,
    )

    robot_pose = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    robot_speed = np.zeros(4, dtype=np.float32)
    plan = np.column_stack(
        [
            np.linspace(0.0, 1.5, 12),
            np.zeros(12),
            np.linspace(0.0, 0.6, 12),
            np.zeros(12),
        ]
    ).astype(np.float32)
    goal = np.array([1.5, 0.0, 0.6, 0.0], dtype=np.float32)
    obstacle_points = np.empty((0, 3), dtype=np.float32)

    command = controller.computeVelocityCommands(
        robot_pose=robot_pose,
        robot_speed=robot_speed,
        plan=plan,
        goal=goal,
        obstacle_points=obstacle_points,
    )

    assert command.shape == (4,)
    assert np.all(np.isfinite(command))
    assert command[0] > 0.0
    assert command[2] > 0.0

    optimal_trajectory = controller.getOptimalTrajectory()
    assert optimal_trajectory.shape == (12, 4)
    assert np.all(np.isfinite(optimal_trajectory))


def test_3d_controller_packs_nearest_controller_obstacle_points_and_masks_padding(
    monkeypatch,
):
    captured = {}

    class FakeOptimizer:
        def __init__(self, *_, **__):
            pass

        def evalControl(self, **kwargs):
            captured["obstacle_points"] = np.asarray(kwargs["obstacle_points"])
            captured["obstacle_points_mask"] = np.asarray(
                kwargs["obstacle_points_mask"]
            )
            return (
                jnp.zeros((4,), dtype=jnp.float32),
                jnp.zeros((1, 4), dtype=jnp.float32),
            )

    monkeypatch.setattr(mppi_3d_controller, "Optimizer3D", FakeOptimizer)
    controller = mppi_3d_controller.MPPIController3D(max_obs_num=3)

    command = controller.computeVelocityCommands(
        robot_pose=np.zeros(4, dtype=np.float32),
        robot_speed=np.zeros(4, dtype=np.float32),
        plan=np.zeros((1, 4), dtype=np.float32),
        goal=np.zeros(4, dtype=np.float32),
        obstacle_points=np.asarray(
            [
                [4.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )

    assert command.shape == (4,)
    np.testing.assert_allclose(captured["obstacle_points_mask"], [1.0, 1.0, 1.0])
    packed_ranges = sorted(
        float(np.dot(point, point)) for point in captured["obstacle_points"]
    )
    np.testing.assert_allclose(packed_ranges, [0.25, 1.0, 4.0])

    controller.computeVelocityCommands(
        robot_pose=np.zeros(4, dtype=np.float32),
        robot_speed=np.zeros(4, dtype=np.float32),
        plan=np.zeros((1, 4), dtype=np.float32),
        goal=np.zeros(4, dtype=np.float32),
        obstacle_points=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
    )

    np.testing.assert_allclose(captured["obstacle_points_mask"], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(captured["obstacle_points"][1:], 0.0)
