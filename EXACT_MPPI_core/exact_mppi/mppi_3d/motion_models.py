import jax
import jax.numpy as jnp

from .models import ControlSequence3D, Trajectories3D


class YawOnly3DHolonomicMotionModel:
    """Kinematic yaw-only 3D holonomic motion model.

    State is [x, y, z, yaw]. Control is [vx, vy, vz, wz], where vx and vy are
    body-frame horizontal velocities, vz is vertical velocity, and wz is yaw
    rate.
    """

    def __init__(self, model_dt: float):
        self.model_dt_ = float(model_dt)

    def integrate_sequence(
        self, control_sequence: ControlSequence3D, pose: jax.Array
    ) -> jax.Array:
        initial_yaw = pose[3]

        def yaw_step(last_yaw, wz_t):
            next_yaw = last_yaw + wz_t * self.model_dt_
            return next_yaw, next_yaw

        _, yaws = jax.lax.scan(yaw_step, initial_yaw, control_sequence.wz)
        yaw_for_velocity = jnp.concatenate([initial_yaw[None], yaws[:-1]], axis=0)

        yaw_cos = jnp.cos(yaw_for_velocity)
        yaw_sin = jnp.sin(yaw_for_velocity)

        dx = control_sequence.vx * yaw_cos - control_sequence.vy * yaw_sin
        dy = control_sequence.vx * yaw_sin + control_sequence.vy * yaw_cos
        dz = control_sequence.vz

        def position_step(carry, delta_t):
            last_x, last_y, last_z = carry
            dx_t, dy_t, dz_t = delta_t
            next_x = last_x + dx_t * self.model_dt_
            next_y = last_y + dy_t * self.model_dt_
            next_z = last_z + dz_t * self.model_dt_
            return (next_x, next_y, next_z), (next_x, next_y, next_z)

        (_, _, _), (xs, ys, zs) = jax.lax.scan(
            position_step,
            (pose[0], pose[1], pose[2]),
            (dx, dy, dz),
        )

        return jnp.stack([xs, ys, zs, yaws], axis=1)

    def integrate_batch(
        self,
        vx: jax.Array,
        vy: jax.Array,
        vz: jax.Array,
        wz: jax.Array,
        pose: jax.Array,
    ) -> Trajectories3D:
        initial_yaw = pose[3]
        batch_size = vx.shape[0]

        def yaw_step(last_yaws, wz_t):
            next_yaws = last_yaws + wz_t * self.model_dt_
            return next_yaws, next_yaws

        init_yaws = jnp.full((batch_size,), initial_yaw)
        _, yaws_t = jax.lax.scan(yaw_step, init_yaws, wz.T)
        yaws = yaws_t.T
        yaw_for_velocity = jnp.concatenate(
            [jnp.full((batch_size, 1), initial_yaw), yaws[:, :-1]], axis=1
        )

        yaw_cos = jnp.cos(yaw_for_velocity)
        yaw_sin = jnp.sin(yaw_for_velocity)

        dx = vx * yaw_cos - vy * yaw_sin
        dy = vx * yaw_sin + vy * yaw_cos

        def position_step(carry, delta_t):
            last_x, last_y, last_z = carry
            dx_t, dy_t, dz_t = delta_t
            next_x = last_x + dx_t * self.model_dt_
            next_y = last_y + dy_t * self.model_dt_
            next_z = last_z + dz_t * self.model_dt_
            return (next_x, next_y, next_z), (next_x, next_y, next_z)

        init_xs = jnp.full((batch_size,), pose[0])
        init_ys = jnp.full((batch_size,), pose[1])
        init_zs = jnp.full((batch_size,), pose[2])
        (_, _, _), (xs_t, ys_t, zs_t) = jax.lax.scan(
            position_step,
            (init_xs, init_ys, init_zs),
            (dx.T, dy.T, vz.T),
        )

        return Trajectories3D(x=xs_t.T, y=ys_t.T, z=zs_t.T, yaws=yaws)
