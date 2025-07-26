import jax
import jax.numpy as jnp
from constants import G, KPC_TO_KM, GYR_TO_S, EPSILON
import functools 

from potentials import NFWAcceleration, PlummerAcceleration
from utils import unwrap_step

### Satellite Functions ###
@jax.jit
def leapfrog_satellite_step(state, dt, logM, Rs, q, dirx, diry, dirz):
    """
    Leapfrog integration step for satellite motion for NFW potential.
    """
    x, y, z, vx, vy, vz = state

    ax, ay, az = NFWAcceleration(x, y, z, logM, Rs, q, dirx, diry, dirz)

    vx_half = vx + 0.5 * dt * ax * KPC_TO_KM**-1
    vy_half = vy + 0.5 * dt * ay * KPC_TO_KM**-1
    vz_half = vz + 0.5 * dt * az * KPC_TO_KM**-1

    x_new = x + dt * vx_half * GYR_TO_S * KPC_TO_KM**-1
    y_new = y + dt * vy_half * GYR_TO_S * KPC_TO_KM**-1
    z_new = z + dt * vz_half * GYR_TO_S * KPC_TO_KM**-1

    ax_new, ay_new, az_new = NFWAcceleration(x_new, y_new, z_new, logM, Rs, q, dirx, diry, dirz)

    vx_new = vx_half + 0.5 * dt * ax_new * KPC_TO_KM**-1
    vy_new = vy_half + 0.5 * dt * ay_new * KPC_TO_KM**-1
    vz_new = vz_half + 0.5 * dt * az_new * KPC_TO_KM**-1

    return (x_new, y_new, z_new, vx_new, vy_new, vz_new)

@functools.partial(jax.jit, static_argnums=(-1,))
def integrate_satellite(x0, y0, z0, vx0, vy0, vz0, logM, Rs, q, dirx, diry, dirz, time, N_STEPS=500):
    """
    Integrates the motion of a satellite using the leapfrog method for NFW potential.
    """
    state = (x0, y0, z0, vx0, vy0, vz0)
    dt    = time/N_STEPS

    # Ensure scalar inputs are JAX arrays
    logM, Rs, q = jnp.asarray(logM), jnp.asarray(Rs), jnp.asarray(q)
    dirx, diry, dirz = jnp.asarray(dirx), jnp.asarray(diry), jnp.asarray(dirz)

    # Step function for JAX scan
    def step_fn(state, _):
        new_state = leapfrog_satellite_step(state, dt, logM, Rs, q, dirx, diry, dirz)
        return new_state, jnp.stack(new_state)  # Ensuring shape consistency

    # Run JAX optimized loop (reverse integration order)
    _, trajectory = jax.lax.scan(step_fn, state, None, length=N_STEPS) #, unroll=True)

    # Ensure trajectory shape is (MAX_LENGHT-1, 6)
    trajectory = jnp.vstack(trajectory)  # Shape: (MAX_LENGHT-1, 6)

    return trajectory

### Stream Functions ###
@jax.jit
def leapfrog_combined_step(state, dt, logM, Rs, q, dirx, diry, dirz, logm, rs):
    """
    Leapfrog integration step for both satellite and stream motion for NFW and Plummer potentials.
    """
    x, y, z, vx, vy, vz, xp, yp, zp, vxp, vyp, vzp = state

    # Update Satellite Position
    axp, ayp, azp = NFWAcceleration(xp, yp, zp, logM, Rs, q, dirx, diry, dirz)

    vxp_half = vxp + 0.5 * dt * axp * KPC_TO_KM**-1
    vyp_half = vyp + 0.5 * dt * ayp * KPC_TO_KM**-1
    vzp_half = vzp + 0.5 * dt * azp * KPC_TO_KM**-1

    xp_new = xp + dt * vxp_half * GYR_TO_S * KPC_TO_KM**-1
    yp_new = yp + dt * vyp_half * GYR_TO_S * KPC_TO_KM**-1
    zp_new = zp + dt * vzp_half * GYR_TO_S * KPC_TO_KM**-1

    axp_new, ayp_new, azp_new = NFWAcceleration(xp_new, yp_new, zp_new, logM, Rs, q, dirx, diry, dirz)

    vxp_new = vxp_half + 0.5 * dt * axp_new * KPC_TO_KM**-1
    vyp_new = vyp_half + 0.5 * dt * ayp_new * KPC_TO_KM**-1
    vzp_new = vzp_half + 0.5 * dt * azp_new * KPC_TO_KM**-1

    # Update Stream Position
    ax, ay, az = NFWAcceleration(x, y, z, logM, Rs, q, dirx, diry, dirz) +  \
                    PlummerAcceleration(x, y, z, logm, rs, x_origin=xp, y_origin=yp, z_origin=zp) # km2 / s / Gyr / kpc

    vx_half = vx + 0.5 * dt * ax * KPC_TO_KM**-1 # km / s
    vy_half = vy + 0.5 * dt * ay * KPC_TO_KM**-1
    vz_half = vz + 0.5 * dt * az * KPC_TO_KM**-1

    x_new = x + dt * vx_half * GYR_TO_S * KPC_TO_KM**-1 # kpc
    y_new = y + dt * vy_half * GYR_TO_S * KPC_TO_KM**-1
    z_new = z + dt * vz_half * GYR_TO_S * KPC_TO_KM**-1

    ax_new, ay_new, az_new = NFWAcceleration(x_new, y_new, z_new, logM, Rs, q, dirx, diry, dirz) +  \
                                PlummerAcceleration(x_new, y_new, z_new, logm, rs, x_origin=xp_new, y_origin=yp_new, z_origin=zp_new) # km2 / s / Gyr / kpc

    vx_new = vx_half + 0.5 * dt * ax_new * KPC_TO_KM**-1 # km / s
    vy_new = vy_half + 0.5 * dt * ay_new * KPC_TO_KM**-1
    vz_new = vz_half + 0.5 * dt * az_new * KPC_TO_KM**-1

    return (x_new, y_new, z_new, vx_new, vy_new, vz_new, xp_new, yp_new, zp_new, vxp_new, vyp_new, vzp_new)

@jax.jit
def integrate_stream_spray(index, x0, y0, z0, vx0, vy0, vz0, theta_sat, xv_sat, logM, Rs, q, dirx, diry, dirz, logm, rs, time, N_STEPS=500):
    # State is a flat tuple of six scalars.
    xp, yp, zp, vxp, vyp, vzp = xv_sat[index]
    thetap = theta_sat[index]
    thetaf = theta_sat[-1]

    theta0 = jnp.arctan2(y0, x0)
    theta0 = jax.lax.cond(theta0 < 0, lambda x: x + 2 * jnp.pi, lambda x: x, theta0)

    state = (theta0, x0, y0, z0, vx0, vy0, vz0, xp, yp, zp, vxp, vyp, vzp)
    dt_sat = time / N_STEPS

    time_here = time - index * dt_sat
    dt_here = time_here / N_STEPS

    def step_fn(state, _):
        # Use only the first three elements of the satellite row.
        theta0, x0, y0, z0, vx0, vy0, vz0, xp, yp, zp, vxp, vyp, vzp = state

        initial_conditions = (x0, y0, z0, vx0, vy0, vz0, xp, yp, zp, vxp, vyp, vzp)
        final_conditions = leapfrog_combined_step(initial_conditions, dt_here,
                                            logM, Rs, q, dirx, diry, dirz, logm, rs)
        
        theta = jnp.arctan2(final_conditions[1], final_conditions[0])
        theta = jax.lax.cond(theta < 0, lambda x: x + 2 * jnp.pi, lambda x: x, theta)

        theta = unwrap_step(theta, theta0)

        new_state = (theta, *final_conditions)

        # The carry and output must have the same structure.
        return new_state, _ # jnp.stack(new_state)

    # Run integration over the satellite trajectory (using all but the last row).
    trajectory, _ = jax.lax.scan(step_fn, state, None, length=N_STEPS) #, unroll=True)
    # 'trajectory' is a tuple of six arrays, each of shape (N_STEPS,).

    theta_count = jnp.floor_divide(thetap, 2 * jnp.pi)
    algin_reference = thetaf - jnp.floor_divide(thetaf, 2 * jnp.pi)*2*jnp.pi # Make sure the angle of reference is at theta=0
    centered_at_0 = (1 - jnp.sign(algin_reference - jnp.pi))/2 * algin_reference + \
                            (1 + jnp.sign(algin_reference - jnp.pi))/2 * (algin_reference - 2 * jnp.pi)

    theta_stream = trajectory[0] - thetaf + theta_count * 2 * jnp.pi + centered_at_0

    return theta_stream, jnp.array(trajectory)[1:7]