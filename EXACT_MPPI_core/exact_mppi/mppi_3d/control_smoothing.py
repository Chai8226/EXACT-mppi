from dataclasses import replace
from typing import Tuple

import jax
import jax.numpy as jnp

from .models import ControlSequence3D, OptimizerSettings3D


def create_control_history_3d(dtype: jnp.dtype = jnp.float32) -> jax.Array:
    return jnp.zeros((4, 4), dtype=dtype)


def process_control_sequence_3d(
    control_sequence: ControlSequence3D,
    control_history: jax.Array,
    settings: OptimizerSettings3D,
) -> Tuple[ControlSequence3D, jax.Array, jax.Array, ControlSequence3D]:
    processed_sequence, updated_history = smooth_control_sequence_with_history_3d(
        control_sequence,
        control_history,
        settings,
    )
    command = select_command_from_sequence_3d(
        processed_sequence,
        settings.shift_control_sequence,
    )
    next_sequence = (
        shift_control_sequence_3d(processed_sequence)
        if settings.shift_control_sequence
        else processed_sequence
    )
    return processed_sequence, updated_history, command, next_sequence


def apply_control_sequence_constraints_3d(
    control_sequence: ControlSequence3D,
    settings: OptimizerSettings3D,
) -> ControlSequence3D:
    constraints = settings.constraints
    vx = limit_linear_axis_delta_3d(
        control_sequence.vx,
        constraints.vx_min,
        constraints.vx_max,
        constraints.ax_min,
        constraints.ax_max,
        settings.model_dt,
    )
    vy = limit_linear_axis_delta_3d(
        control_sequence.vy,
        -constraints.vy,
        constraints.vy,
        constraints.ay_min,
        constraints.ay_max,
        settings.model_dt,
    )
    vz = limit_linear_axis_delta_3d(
        control_sequence.vz,
        -constraints.vz,
        constraints.vz,
        constraints.az_min,
        constraints.az_max,
        settings.model_dt,
    )
    wz = limit_symmetric_axis_delta_3d(
        control_sequence.wz,
        constraints.wz,
        constraints.awz_max,
        settings.model_dt,
    )
    return replace(control_sequence, vx=vx, vy=vy, vz=vz, wz=wz)


def limit_linear_axis_delta_3d(
    values: jax.Array,
    lower: float,
    upper: float,
    accel_min: float,
    accel_max: float,
    model_dt: float,
) -> jax.Array:
    max_delta = model_dt * accel_max
    min_delta = model_dt * accel_min
    first = jnp.clip(values[0], lower, upper)

    def step(last, current):
        current = jnp.clip(current, lower, upper)
        lo = jnp.where(last > 0.0, last + min_delta, last - max_delta)
        hi = jnp.where(last > 0.0, last + max_delta, last - min_delta)
        current = jnp.clip(current, lo, hi)
        return current, current

    _, rest = jax.lax.scan(step, first, values[1:])
    return jnp.concatenate([first[None], rest], axis=0)


def limit_symmetric_axis_delta_3d(
    values: jax.Array,
    limit: float,
    accel_limit: float,
    model_dt: float,
) -> jax.Array:
    max_delta = model_dt * accel_limit
    first = jnp.clip(values[0], -limit, limit)

    def step(last, current):
        current = jnp.clip(current, -limit, limit)
        current = jnp.clip(current, last - max_delta, last + max_delta)
        return current, current

    _, rest = jax.lax.scan(step, first, values[1:])
    return jnp.concatenate([first[None], rest], axis=0)


def smooth_control_sequence_with_history_3d(
    control_sequence: ControlSequence3D,
    control_history: jax.Array,
    settings: OptimizerSettings3D,
) -> Tuple[ControlSequence3D, jax.Array]:
    num_sequences = control_sequence.vx.shape[0] - 1

    def do_nothing(_):
        return control_sequence, control_history

    def apply_filter(_):
        filtered = savitzky_golay_filter_control_sequence_3d(
            control_sequence,
            control_history,
        )
        filtered = apply_control_sequence_constraints_3d(filtered, settings)
        new_control = select_command_from_sequence_3d(
            filtered,
            settings.shift_control_sequence,
        )
        updated_history = jnp.concatenate(
            [control_history[1:], new_control[None, :]],
            axis=0,
        )
        return filtered, updated_history

    return jax.lax.cond(num_sequences < 4, do_nothing, apply_filter, operand=None)


def savitzky_golay_filter_control_sequence_3d(
    control_sequence: ControlSequence3D,
    control_history: jax.Array,
) -> ControlSequence3D:
    num_sequences = control_sequence.vx.shape[0] - 1
    use_nine_point = num_sequences >= 20

    def filter_axis(sequence: jax.Array, history: jax.Array) -> jax.Array:
        def nine_point(_):
            coeffs = (
                jnp.array(
                    [-21.0, 14.0, 39.0, 54.0, 59.0, 54.0, 39.0, 14.0, -21.0],
                    dtype=jnp.float32,
                )
                / 231.0
            )
            padded = jnp.concatenate(
                [history, sequence, jnp.full((4,), sequence[-1])],
                axis=0,
            )

            def apply_at(idx):
                data = jax.lax.dynamic_slice(padded, (idx,), (9,))
                return jnp.sum(data * coeffs)

            idxs = jnp.arange(num_sequences, dtype=jnp.int32)
            return sequence.at[:num_sequences].set(jax.vmap(apply_at)(idxs))

        def five_point(_):
            coeffs = (
                jnp.array([-3.0, 12.0, 17.0, 12.0, -3.0], dtype=jnp.float32)
                / 35.0
            )
            padded = jnp.concatenate(
                [history[-2:], sequence, jnp.full((2,), sequence[-1])],
                axis=0,
            )

            def apply_at(idx):
                data = jax.lax.dynamic_slice(padded, (idx,), (5,))
                return jnp.sum(data * coeffs)

            idxs = jnp.arange(num_sequences, dtype=jnp.int32)
            return sequence.at[:num_sequences].set(jax.vmap(apply_at)(idxs))

        return jax.lax.cond(use_nine_point, nine_point, five_point, operand=None)

    return ControlSequence3D(
        vx=filter_axis(control_sequence.vx, control_history[:, 0]),
        vy=filter_axis(control_sequence.vy, control_history[:, 1]),
        vz=filter_axis(control_sequence.vz, control_history[:, 2]),
        wz=filter_axis(control_sequence.wz, control_history[:, 3]),
    )


def select_command_from_sequence_3d(
    control_sequence: ControlSequence3D,
    shift_control_sequence: bool,
) -> jax.Array:
    shift_control_sequence_ = jnp.asarray(shift_control_sequence, dtype=jnp.bool_)
    offset = jnp.where(shift_control_sequence_, jnp.int32(1), jnp.int32(0))
    return jnp.stack(
        [
            control_sequence.vx[offset],
            control_sequence.vy[offset],
            control_sequence.vz[offset],
            control_sequence.wz[offset],
        ],
        axis=0,
    )


def shift_control_sequence_3d(control_sequence: ControlSequence3D) -> ControlSequence3D:
    return ControlSequence3D(
        vx=jnp.concatenate([control_sequence.vx[1:], control_sequence.vx[-1:]]),
        vy=jnp.concatenate([control_sequence.vy[1:], control_sequence.vy[-1:]]),
        vz=jnp.concatenate([control_sequence.vz[1:], control_sequence.vz[-1:]]),
        wz=jnp.concatenate([control_sequence.wz[1:], control_sequence.wz[-1:]]),
    )
