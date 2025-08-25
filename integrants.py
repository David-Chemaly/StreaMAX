import jax
import jax.numpy as jnp
from constants import G, KPC_TO_KM, GYR_TO_S, EPSILON
import functools 

from potentials import NFWAcceleration, NFWHessian, PlummerAcceleration, PlummerHessian
from utils import unwrap_step, update_streams

### Stream Functions ###

def leapfrog_first_combined_step(state, dt, logM, Rs, q, dirx, diry, dirz, logm, rs):
    """
    Leapfrog integration step for both satellite and stream motion for NFW and Plummer potentials.
    """
    (x, y, z, vx, vy, vz, xp, yp, zp, vxp, vyp, vzp), S, dS = state

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

    # Update first degree
    Hess_old = NFWHessian(x, y, z, logM, Rs, q, dirx, diry, dirz) * KPC_TO_KM**-2 * GYR_TO_S**2  +  \
                    PlummerHessian(x, y, z, logm, rs, x_origin=xp, y_origin=yp, z_origin=zp) * KPC_TO_KM**-2 * GYR_TO_S**2 # km2 / s / Gyr / kpc2 -> 1 / Gyr2
    ddS = Hess_old @ S 
    dS_half = dS + 0.5 * dt * ddS  # 1 / Gyr
    S_new = S + dt * dS_half  # back to S units

    Hess_new = NFWHessian(x_new, y_new, z_new, logM, Rs, q, dirx, diry, dirz) * KPC_TO_KM**-2 * GYR_TO_S**2  +  \
                    PlummerHessian(x_new, y_new, z_new, logm, rs, x_origin=xp_new, y_origin=yp_new, z_origin=zp_new) * KPC_TO_KM**-2 * GYR_TO_S**2 # km2 / s / Gyr / kpc2 -> 1 / Gyr2
    ddS_new = Hess_new @ S_new
    dS_new = dS_half + 0.5 * dt * ddS_new  # 1 / Gyr

    # TODO: update second degree

    return (x_new, y_new, z_new, vx_new, vy_new, vz_new, xp_new, yp_new, zp_new, vxp_new, vyp_new, vzp_new), S_new, dS_new #, vS_new



@functools.partial(jax.jit, static_argnums=(-1,))
def integrate_stream_first(index, x0, y0, z0, vx0, vy0, vz0, theta_sat, xv_sat, logM, Rs, q, dirx, diry, dirz, logm, rs, time, N_STEPS=500):
    # State is a flat tuple of six scalars.
    xp, yp, zp, vxp, vyp, vzp = xv_sat[index]
    thetap = theta_sat[index]
    thetaf = theta_sat[-1]

    theta0 = jnp.arctan2(y0, x0)
    theta0 = jax.lax.cond(theta0 < 0, lambda x: x + 2 * jnp.pi, lambda x: x, theta0)

    S = jnp.block([jnp.eye(3), jnp.zeros([3,3])])  # Initial state for the stream, identity matrix for the Hessian
    dS = jnp.zeros([3,6])

    state = ((theta0, x0, y0, z0, vx0, vy0, vz0, xp, yp, zp, vxp, vyp, vzp), S, dS)
    dt_sat = time / N_STEPS

    time_here = time - index * dt_sat
    dt_here = time_here / N_STEPS

    def step_fn(state, _):
        # Use only the first three elements of the satellite row.
        (theta0, x0, y0, z0, vx0, vy0, vz0, xp, yp, zp, vxp, vyp, vzp), S, dS = state

        initial_conditions = ((x0, y0, z0, vx0, vy0, vz0, xp, yp, zp, vxp, vyp, vzp), S, dS)
        final_conditions = leapfrog_first_combined_step(initial_conditions, dt_here,
                                            logM, Rs, q, dirx, diry, dirz, logm, rs)

        theta = jnp.arctan2(final_conditions[0][1], final_conditions[0][0])
        theta = jax.lax.cond(theta < 0, lambda x: x + 2 * jnp.pi, lambda x: x, theta)

        theta = unwrap_step(theta, theta0)

        new_state = ((theta, *final_conditions[0]), *final_conditions[1:])

        # The carry and output must have the same structure.
        return new_state, _ # jnp.stack(new_state)

    # Run integration over the satellite trajectory (using all but the last row).
    trajectory, _ = jax.lax.scan(step_fn, state, None, length=N_STEPS) #, unroll=True)
    # 'trajectory' is a tuple of six arrays, each of shape (N_STEPS,).

    theta_count = jnp.floor_divide(thetap, 2 * jnp.pi)
    algin_reference = thetaf - jnp.floor_divide(thetaf, 2 * jnp.pi)*2*jnp.pi # Make sure the angle of reference is at theta=0
    centered_at_0 = (1 - jnp.sign(algin_reference - jnp.pi))/2 * algin_reference + \
                            (1 + jnp.sign(algin_reference - jnp.pi))/2 * (algin_reference - 2 * jnp.pi)

    theta_stream = trajectory[0][0] - thetaf + theta_count * 2 * jnp.pi + centered_at_0

    return theta_stream, jnp.array(trajectory[0])[1:7], jnp.array(trajectory[1]), jnp.array(trajectory[2])  # Return the stream state and Hessian

@jax.jit
def integrate_stream_streak(index, x0, y0, z0, vx0, vy0, vz0, theta_sat, xv_sat, logM, Rs, q, dirx, diry, dirz, logm, rs, time, N_STEPS=500, buffer_size=10):
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

    def step_fn(carry, _):
        buffer, state = carry
        theta0, x0, y0, z0, vx0, vy0, vz0, xp, yp, zp, vxp, vyp, vzp = state

        initial_conditions = (x0, y0, z0, vx0, vy0, vz0, xp, yp, zp, vxp, vyp, vzp)
        final_conditions = leapfrog_combined_step(initial_conditions, dt_here,
                                                logM, Rs, q, dirx, diry, dirz, logm, rs)

        theta = jnp.arctan2(final_conditions[1], final_conditions[0])
        theta = jax.lax.cond(theta < 0, lambda x: x + 2 * jnp.pi, lambda x: x, theta)
        theta = unwrap_step(theta, theta0)
        new_state = (theta, *final_conditions)

        # Update buffer: roll left and append new_state
        new_buffer = jnp.roll(buffer, -1, axis=0).at[-1].set(jnp.array(new_state))
        return (new_buffer, new_state), new_buffer

    # Initialize buffer with the initial state repeated
    buffer_init = jnp.tile(jnp.array(state)[None, :], (buffer_size, 1))
    (final_buffer, _), buffers = jax.lax.scan(step_fn, (buffer_init, state), None, length=N_STEPS + (buffer_size//2))

    theta_count = jnp.floor_divide(thetap, 2 * jnp.pi)
    algin_reference = thetaf - jnp.floor_divide(thetaf, 2 * jnp.pi)*2*jnp.pi # Make sure the angle of reference is at theta=0
    centered_at_0 = (1 - jnp.sign(algin_reference - jnp.pi))/2 * algin_reference + \
                            (1 + jnp.sign(algin_reference - jnp.pi))/2 * (algin_reference - 2 * jnp.pi)
    theta_stream = final_buffer[:, 0] - thetaf + theta_count * 2 * jnp.pi + centered_at_0

    return theta_stream, final_buffer[:, 1:7] # Flatten the first dimension to get a 2D array

### Binned ###
@jax.jit
def leapfrog_individual_step(count, theta0, state, dt, logM, Rs, q, dirx, diry, dirz, logm, rs, xp, yp, zp, vxp, vyp, vzp, r_threshold):
    """
    Leapfrog integration step for both satellite and stream motion for NFW and Plummer potentials.
    """
    x, y, z, vx, vy, vz = state

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

    theta = jnp.arctan2(y_new, x_new)
    theta = jax.lax.cond(theta < 0, lambda x: x + 2 * jnp.pi, lambda x: x, theta)
    theta = unwrap_step(theta, theta0)

    r_dist_here = jnp.sqrt( (x_new - xp_new)**2 + 
                            (y_new - yp_new)**2 + 
                            (z_new - zp_new)**2 )

    mask  = jax.lax.cond( (r_dist_here > r_threshold) & (count == 0), lambda: 1., lambda: jnp.nan)

    vt_here = jnp.sqrt( (vx_new - vxp_new)**2 + 
                        (vy_new - vyp_new)**2 + 
                        (vz_new - vzp_new)**2 )

    return theta, jnp.array([x_new, y_new, z_new, vx_new, vy_new, vz_new]), mask, vt_here

@jax.jit
def masked_leapfrog(count_j, theta0_j, xv_j, idx_j, current_i, fwd_i_args, r_threshold, dt, logM, Rs, q, dirx, diry, dirz, logm, rs):
    def do_step(_):
        return leapfrog_individual_step(count_j, theta0_j, xv_j, dt, logM, Rs, q, dirx, diry, dirz, logm, rs, *fwd_i_args, r_threshold)

    def skip_step(_):
        return theta0_j, xv_j, jnp.nan, jnp.nan

    return jax.lax.cond( (idx_j <= current_i) & (~jnp.isnan(count_j)), do_step, skip_step, operand=None)

@functools.partial(jax.jit, static_argnums=(1, 2))
def wrapper_scan(carry_init, n_steps, num_bin, forward_trajectory, index, r_threshold, dt, logM, Rs, q, dirx, diry, dirz, logm, rs):

    @jax.jit
    def scan_step(carry, i):
        count_stream, theta_stream, xv_stream = carry

        # You need to compute forward_trajectory[i] outside `scan` and pass it in
        fwd_i = forward_trajectory[i]

        # Vectorized leapfrog step
        theta_stream, xv_stream, mask, vt = jax.vmap(
            masked_leapfrog, in_axes=(0, 0, 0, 0, None, None, None, None, None, None, None, None, None, None, None, None)
        )(count_stream, theta_stream, xv_stream, index, i, fwd_i, r_threshold, dt, logM, Rs, q, dirx, diry, dirz, logm, rs)

        wrong_bool_mask = jnp.isnan(mask)
        bool_mask = ~wrong_bool_mask
        arg_remove = wrong_bool_mask * 1.0 + bool_mask * jnp.nan

        nb_out = jnp.nansum(mask)

        count_stream, theta_stream, xv_stream = update_streams(
            count_stream, theta_stream, xv_stream,
            vt, mask, arg_remove, nb_out, bool_mask, num_bin
        )

        return (count_stream, theta_stream, xv_stream), None
    
    return jax.lax.scan(scan_step, carry_init, jnp.arange(n_steps))

# @functools.partial(jax.jit, static_argnums=(1, 2))
# def wrapper_scan(carry_init, n_steps, num_bin):
#     def scan_body(carry, i):
#         return scan_step(carry, i, num_bin)
#     return jax.lax.scan(scan_body, carry_init, jnp.arange(n_steps))