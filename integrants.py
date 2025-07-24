import jax
import jax.numpy as jnp
from constants import G, KPC_TO_KM, GYR_TO_S, EPSILON
import functools 

from potentials import NFWAcceleration, PlummerAcceleration

### Satellite Functions ###
@functools.partial(jax.jit, static_argnums=(-1,))
def leapfrog_satellite_step(state, dt, logM, Rs, q, dirx, diry, dirz, acc_func=NFWAcceleration):
    x, y, z, vx, vy, vz = state

    ax, ay, az = acc_func(x, y, z, logM, Rs, q, dirx, diry, dirz)

    vx_half = vx + 0.5 * dt * ax * KPC_TO_KM**-1
    vy_half = vy + 0.5 * dt * ay * KPC_TO_KM**-1
    vz_half = vz + 0.5 * dt * az * KPC_TO_KM**-1

    x_new = x + dt * vx_half * GYR_TO_S * KPC_TO_KM**-1
    y_new = y + dt * vy_half * GYR_TO_S * KPC_TO_KM**-1
    z_new = z + dt * vz_half * GYR_TO_S * KPC_TO_KM**-1

    ax_new, ay_new, az_new = acc_func(x_new, y_new, z_new, logM, Rs, q, dirx, diry, dirz)

    vx_new = vx_half + 0.5 * dt * ax_new * KPC_TO_KM**-1
    vy_new = vy_half + 0.5 * dt * ay_new * KPC_TO_KM**-1
    vz_new = vz_half + 0.5 * dt * az_new * KPC_TO_KM**-1

    return (x_new, y_new, z_new, vx_new, vy_new, vz_new)

@functools.partial(jax.jit, static_argnums=(-2, -1, ))
def integrate_satellite(x0, y0, z0, vx0, vy0, vz0, logM, Rs, q, dirx, diry, dirz, time, integrate_func=leapfrog_satellite_step, N_STEPS=100):
    state = (x0, y0, z0, vx0, vy0, vz0)
    dt    = time/N_STEPS

    # Ensure scalar inputs are JAX arrays
    logM, Rs, q = jnp.asarray(logM), jnp.asarray(Rs), jnp.asarray(q)
    dirx, diry, dirz = jnp.asarray(dirx), jnp.asarray(diry), jnp.asarray(dirz)

    # Step function for JAX scan
    def step_fn(state, _):
        new_state = integrate_func(state, dt, logM, Rs, q, dirx, diry, dirz)
        return new_state, jnp.stack(new_state)  # Ensuring shape consistency

    # Run JAX optimized loop (reverse integration order)
    _, trajectory = jax.lax.scan(step_fn, state, None, length=N_STEPS - 1, unroll=True)

    # Ensure trajectory shape is (MAX_LENGHT-1, 6)
    trajectory = jnp.array(trajectory)  # Shape: (MAX_LENGHT-1, 6)

    # Correct concatenation
    trajectory = jnp.vstack([trajectory[::-1], jnp.array(state)[None, :]])  # Shape: (MAX_LENGHT, 6)

    # Compute time steps
    time_steps = jnp.arange(N_STEPS) * dt

    return trajectory, time_steps

### Stream Functions ###
@functools.partial(jax.jit, static_argnums=(-2, -1))
def leapfrog_combined_step(state, dt, logM, Rs, q, dirx, diry, dirz, logm, rs, acc_func1=NFWAcceleration, acc_func2=PlummerAcceleration):
    x, y, z, vx, vy, vz, xp, yp, zp, vxp, vyp, vzp = state

    # Update Satellite Position
    axp, ayp, azp = acc_func1(xp, yp, zp, logM, Rs, q, dirx, diry, dirz)

    vxp_half = vxp + 0.5 * dt * axp * KPC_TO_KM**-1
    vyp_half = vyp + 0.5 * dt * ayp * KPC_TO_KM**-1
    vzp_half = vzp + 0.5 * dt * azp * KPC_TO_KM**-1

    xp_new = xp + dt * vxp_half * GYR_TO_S * KPC_TO_KM**-1
    yp_new = yp + dt * vyp_half * GYR_TO_S * KPC_TO_KM**-1
    zp_new = zp + dt * vzp_half * GYR_TO_S * KPC_TO_KM**-1

    axp_new, ayp_new, azp_new = acc_func1(xp_new, yp_new, zp_new, logM, Rs, q, dirx, diry, dirz)

    vxp_new = vxp_half + 0.5 * dt * axp_new * KPC_TO_KM**-1
    vyp_new = vyp_half + 0.5 * dt * ayp_new * KPC_TO_KM**-1
    vzp_new = vzp_half + 0.5 * dt * azp_new * KPC_TO_KM**-1

    # Update Stream Position
    ax, ay, az = acc_func1(x, y, z, logM, Rs, q, dirx, diry, dirz) +  \
                    acc_func2(x, y, z, logm, rs, x_origin=xp, y_origin=yp, z_origin=zp) # km2 / s / Gyr / kpc

    vx_half = vx + 0.5 * dt * ax * KPC_TO_KM**-1 # km / s
    vy_half = vy + 0.5 * dt * ay * KPC_TO_KM**-1
    vz_half = vz + 0.5 * dt * az * KPC_TO_KM**-1

    x_new = x + dt * vx_half * GYR_TO_S * KPC_TO_KM**-1 # kpc
    y_new = y + dt * vy_half * GYR_TO_S * KPC_TO_KM**-1
    z_new = z + dt * vz_half * GYR_TO_S * KPC_TO_KM**-1

    ax_new, ay_new, az_new = acc_func1(x_new, y_new, z_new, logM, Rs, q, dirx, diry, dirz) +  \
                                acc_func2(x_new, y_new, z_new, logm, rs, x_origin=xp_new, y_origin=yp_new, z_origin=zp_new) # km2 / s / Gyr / kpc

    vx_new = vx_half + 0.5 * dt * ax_new * KPC_TO_KM**-1 # km / s
    vy_new = vy_half + 0.5 * dt * ay_new * KPC_TO_KM**-1
    vz_new = vz_half + 0.5 * dt * az_new * KPC_TO_KM**-1

    return (x_new, y_new, z_new, vx_new, vy_new, vz_new, xp_new, yp_new, zp_new, vxp_new, vyp_new, vzp_new)