import jax.numpy as jnp
import numpy as np

from exact_mppi.mppi_3d.control_smoothing import (
    create_control_history_3d,
    limit_linear_axis_delta_3d,
    process_control_sequence_3d,
)
from exact_mppi.mppi_3d.models import (
    ControlConstraints3D,
    ControlSequence3D,
    OptimizerSettings3D,
    SamplingStd3D,
)


def _settings(
    *,
    shift_control_sequence: bool = True,
    ax_max: float = 100.0,
    ax_min: float = -100.0,
) -> OptimizerSettings3D:
    return OptimizerSettings3D(
        constraints=ControlConstraints3D(
            vx_max=20.0,
            vx_min=-20.0,
            vy=20.0,
            vz=20.0,
            wz=20.0,
            ax_max=ax_max,
            ax_min=ax_min,
            ay_max=100.0,
            ay_min=-100.0,
            az_max=100.0,
            az_min=-100.0,
            awz_max=100.0,
        ),
        sampling_std=SamplingStd3D(vx=0.1, vy=0.1, vz=0.1, wz=0.1),
        model_dt=0.1,
        temperature=0.3,
        batch_size=4,
        time_steps=6,
        iteration_count=1,
        shift_control_sequence=shift_control_sequence,
        goal_weight=1.0,
        goal_yaw_weight=1.0,
        path_weight=1.0,
        control_weight=1.0,
    )


def test_process_control_sequence_3d_filters_selects_command_updates_history_and_shifts():
    sequence = ControlSequence3D(
        vx=jnp.asarray([1.0, 2.0, 4.0, 7.0, 11.0, 16.0], dtype=jnp.float32),
        vy=jnp.asarray([0.0, -1.0, -1.0, 0.0, 2.0, 5.0], dtype=jnp.float32),
        vz=jnp.asarray([0.0, 0.5, 1.5, 3.0, 5.0, 7.5], dtype=jnp.float32),
        wz=jnp.asarray([0.0, 0.2, 0.6, 1.2, 2.0, 3.0], dtype=jnp.float32),
    )
    history = jnp.asarray(
        [
            [-1.0, 0.5, -0.5, -0.2],
            [-0.5, 0.25, -0.25, -0.1],
            [0.0, 0.0, 0.0, 0.0],
            [0.5, -0.25, 0.25, 0.1],
        ],
        dtype=jnp.float32,
    )

    processed, updated_history, command, next_sequence = process_control_sequence_3d(
        sequence,
        history,
        _settings(shift_control_sequence=True),
    )

    np.testing.assert_allclose(
        np.asarray(processed.vx),
        [1.0, 2.0428572, 4.0, 7.0, 11.514286, 16.0],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(command),
        [2.0428572, -0.80714285, 0.47857144, 0.19142857],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(updated_history),
        [
            [-0.5, 0.25, -0.25, -0.1],
            [0.0, 0.0, 0.0, 0.0],
            [0.5, -0.25, 0.25, 0.1],
            [2.0428572, -0.80714285, 0.47857144, 0.19142857],
        ],
        rtol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(next_sequence.vx[:2]), [2.0428572, 4.0])
    np.testing.assert_allclose(np.asarray(next_sequence.vx[-1]), 16.0)


def test_process_control_sequence_3d_keeps_short_sequences_unfiltered():
    sequence = ControlSequence3D(
        vx=jnp.asarray([0.1, 0.2, 0.3, 0.4], dtype=jnp.float32),
        vy=jnp.asarray([0.0, 0.1, 0.2, 0.3], dtype=jnp.float32),
        vz=jnp.asarray([0.5, 0.6, 0.7, 0.8], dtype=jnp.float32),
        wz=jnp.asarray([-0.1, -0.2, -0.3, -0.4], dtype=jnp.float32),
    )
    history = create_control_history_3d()

    processed, updated_history, command, next_sequence = process_control_sequence_3d(
        sequence,
        history,
        _settings(shift_control_sequence=False),
    )

    np.testing.assert_allclose(np.asarray(processed.vx), [0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(np.asarray(updated_history), np.zeros((4, 4)))
    np.testing.assert_allclose(np.asarray(command), [0.1, 0.0, 0.5, -0.1])
    np.testing.assert_allclose(np.asarray(next_sequence.vx), [0.1, 0.2, 0.3, 0.4])


def test_linear_delta_limit_uses_accel_safety_rule_across_zero():
    limited = limit_linear_axis_delta_3d(
        jnp.asarray([0.2, -1.0, -2.0, 2.0], dtype=jnp.float32),
        lower=-10.0,
        upper=10.0,
        accel_min=-1.0,
        accel_max=3.0,
        model_dt=0.1,
    )

    np.testing.assert_allclose(
        np.asarray(limited),
        [0.2, 0.1, 0.0, 0.1],
        rtol=1e-6,
        atol=1e-6,
    )
