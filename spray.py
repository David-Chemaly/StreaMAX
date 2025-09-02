import jax
import jax.numpy as jnp
import functools

from utils import unwrap_step
from potentials import NFWAcceleration, PlummerAcceleration, NFWHessian
from utils import jax_unwrap, get_rj_vj_R
from constants import KMS_TO_KPCGYR, KPCGYR_TO_KMS, TWOPI

N_STEPS = 100
N_PARTICLES = 1000

### Satellite Functions ###
@jax.jit
def leapfrog_satellite_step(state, dt, logM, Rs, q, dirx, diry, dirz):
    """
    Leapfrog integration step for satellite motion for NFW potential.
    """
    x, y, z, vx, vy, vz = state

    ax, ay, az = NFWAcceleration(x, y, z, logM, Rs, q, dirx, diry, dirz)

    vx_half = vx + 0.5 * dt * ax
    vy_half = vy + 0.5 * dt * ay
    vz_half = vz + 0.5 * dt * az

    x_new = x + dt * vx_half
    y_new = y + dt * vy_half
    z_new = z + dt * vz_half

    ax_new, ay_new, az_new = NFWAcceleration(x_new, y_new, z_new, logM, Rs, q, dirx, diry, dirz)

    vx_new = vx_half + 0.5 * dt * ax_new
    vy_new = vy_half + 0.5 * dt * ay_new
    vz_new = vz_half + 0.5 * dt * az_new

    return (x_new, y_new, z_new, vx_new, vy_new, vz_new)

@jax.jit
def integrate_satellite(x0, y0, z0, vx0, vy0, vz0, logM, Rs, q, dirx, diry, dirz, time):
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
    _, trajectory = jax.lax.scan(step_fn, state, None, length=N_STEPS, unroll=True)

    # Ensure trajectory shape is (MAX_LENGHT-1, 6)
    trajectory = jnp.vstack(trajectory)  # Shape: (MAX_LENGHT-1, 6)

    return trajectory

@jax.jit
def leapfrog_combined_step(state, dt, logM, Rs, q, dirx, diry, dirz, logm, rs):
    """
    Leapfrog integration step for both satellite and stream motion for NFW and Plummer potentials.
    """
    x, y, z, vx, vy, vz, xp, yp, zp, vxp, vyp, vzp = state

    # Update Satellite Position
    axp, ayp, azp = NFWAcceleration(xp, yp, zp, logM, Rs, q, dirx, diry, dirz)

    vxp_half = vxp + 0.5 * dt * axp
    vyp_half = vyp + 0.5 * dt * ayp
    vzp_half = vzp + 0.5 * dt * azp

    xp_new = xp + dt * vxp_half
    yp_new = yp + dt * vyp_half
    zp_new = zp + dt * vzp_half

    axp_new, ayp_new, azp_new = NFWAcceleration(xp_new, yp_new, zp_new, logM, Rs, q, dirx, diry, dirz)

    vxp_new = vxp_half + 0.5 * dt * axp_new
    vyp_new = vyp_half + 0.5 * dt * ayp_new
    vzp_new = vzp_half + 0.5 * dt * azp_new

    # Update Stream Position
    ax, ay, az = NFWAcceleration(x, y, z, logM, Rs, q, dirx, diry, dirz) +  \
                    PlummerAcceleration(x, y, z, logm, rs, x_origin=xp, y_origin=yp, z_origin=zp)

    vx_half = vx + 0.5 * dt * ax
    vy_half = vy + 0.5 * dt * ay
    vz_half = vz + 0.5 * dt * az

    x_new = x + dt * vx_half
    y_new = y + dt * vy_half
    z_new = z + dt * vz_half

    ax_new, ay_new, az_new = NFWAcceleration(x_new, y_new, z_new, logM, Rs, q, dirx, diry, dirz) +  \
                                PlummerAcceleration(x_new, y_new, z_new, logm, rs, x_origin=xp_new, y_origin=yp_new, z_origin=zp_new)

    vx_new = vx_half + 0.5 * dt * ax_new
    vy_new = vy_half + 0.5 * dt * ay_new
    vz_new = vz_half + 0.5 * dt * az_new

    return (x_new, y_new, z_new, vx_new, vy_new, vz_new, xp_new, yp_new, zp_new, vxp_new, vyp_new, vzp_new)

@jax.jit
def integrate_stream_spray(index, x0, y0, z0, vx0, vy0, vz0, theta_sat, xv_sat, logM, Rs, q, dirx, diry, dirz, logm, rs, time):
    # State is a flat tuple of six scalars.
    xp, yp, zp, vxp, vyp, vzp = xv_sat[index]
    thetap = theta_sat[index]
    thetaf = theta_sat[-1]

    theta0 = jnp.arctan2(y0, x0)
    theta0 = jax.lax.cond(theta0 < 0, lambda x: x + TWOPI, lambda x: x, theta0)

    state = (theta0, x0, y0, z0, vx0, vy0, vz0, xp, yp, zp, vxp, vyp, vzp)
    dt_sat = time / N_STEPS

    time_here = time - index * dt_sat
    dt_here = time_here / N_STEPS

    def step_fn(state, _):
        # Use only the first three elements of the satellite row.
        theta0, x0, y0, z0, vx0, vy0, vz0, xp, yp, zp, vxp, vyp, vzp = state

        initial_conditions = (x0, y0, z0, vx0, vy0, vz0, xp, yp, zp, vxp, vyp, vzp)
        final_conditions   = leapfrog_combined_step(initial_conditions, dt_here,
                                            logM, Rs, q, dirx, diry, dirz, logm, rs)
        
        theta = jnp.arctan2(final_conditions[1], final_conditions[0])
        theta = jax.lax.cond(theta < 0, lambda x: x + TWOPI, lambda x: x, theta)

        theta = unwrap_step(theta, theta0)

        new_state = (theta, *final_conditions)

        # The carry and output must have the same structure.
        return new_state, _ # jnp.stack(new_state)

    # Run integration over the satellite trajectory (using all but the last row).
    trajectory, _ = jax.lax.scan(step_fn, state, None, length=N_STEPS, unroll=True)
    # 'trajectory' is a tuple of six arrays, each of shape (N_STEPS,).

    thetap_bound = thetap - TWOPI*jnp.floor_divide(thetap, TWOPI)
    theta_diff = jnp.abs(thetap_bound - theta0)
    sign = jnp.sign(jnp.cross(jnp.array([xp, yp]), jnp.array([x0, y0])))
    theta_count = jnp.floor_divide(thetap + sign * theta_diff, TWOPI)

    algin_reference = thetaf - jnp.floor_divide(thetaf, TWOPI)*TWOPI # Make sure the angle of reference is at theta=0
    centered_at_0 = (1 - jnp.sign(algin_reference - jnp.pi))/2 * algin_reference + \
                            (1 + jnp.sign(algin_reference - jnp.pi))/2 * (algin_reference - TWOPI)

    theta_stream = trajectory[0] - thetaf + theta_count * TWOPI + centered_at_0

    return theta_stream, jnp.array(trajectory)[1:7]

@jax.jit
def create_ic_particle_spray(orbit_sat, rj, vj, R, tail=0, seed=111):
    key=jax.random.PRNGKey(seed)
    N = rj.shape[0]

    tile = jax.lax.cond(tail == 0, lambda _: jnp.tile(jnp.array([1, -1]), N_PARTICLES//2),
                        lambda _: jax.lax.cond(tail == 1, lambda _: jnp.tile(jnp.array([-1, -1]), N_PARTICLES//2),
                        lambda _: jnp.tile(jnp.array([1, 1]), N_PARTICLES//2), None), None)

    rj = jnp.repeat(rj, N_PARTICLES//N_STEPS) * tile
    vj = jnp.repeat(vj, N_PARTICLES//N_STEPS) * tile
    R  = jnp.repeat(R, N_PARTICLES//N_STEPS, axis=0)  # Shape: (2N, 3, 3)

    # Parameters for position and velocity offsets
    mean_x, disp_x = 2.0, 0.5
    disp_z = 0.5
    mean_vy, disp_vy = 0.3, 0.5
    disp_vz = 0.5

    # Generate random samples for position and velocity offsets
    key, subkey_x, subkey_z, subkey_vy, subkey_vz = jax.random.split(key, 5)
    rx = jax.random.normal(subkey_x, shape=(N_PARTICLES//N_STEPS * N,)) * disp_x + mean_x
    rz = jax.random.normal(subkey_z, shape=(N_PARTICLES//N_STEPS * N,)) * disp_z * rj
    rvy = (jax.random.normal(subkey_vy, shape=(N_PARTICLES//N_STEPS * N,)) * disp_vy + mean_vy) * vj * rx
    rvz = jax.random.normal(subkey_vz, shape=(N_PARTICLES//N_STEPS * N,)) * disp_vz * vj
    rx *= rj  # Scale x displacement by rj

    # Position and velocity offsets in the satellite reference frame
    offset_pos = jnp.column_stack([rx, jnp.zeros_like(rx), rz])  # Shape: (2N, 3)
    offset_vel = jnp.column_stack([jnp.zeros_like(rx), rvy, rvz])  # Shape: (2N, 3)

    # Transform to the host-centered frame
    orbit_sat_repeated = jnp.repeat(orbit_sat, N_PARTICLES//N_STEPS, axis=0)  # More efficient than tile+reshape
    offset_pos_transformed = jnp.einsum('ni,nij->nj', offset_pos, R)
    offset_vel_transformed = jnp.einsum('ni,nij->nj', offset_vel, R)

    ic_stream = orbit_sat_repeated + jnp.concatenate([offset_pos_transformed, offset_vel_transformed], axis=-1)

    return ic_stream  # Shape: (N_particule, 6)

@jax.jit
def generate_stream_spray(params,  seed, tail=0):
    """
    Generates a stream spray based on the provided parameters and integrates the satellite motion.
    """
    logM, Rs, q, dirx, diry, dirz, logm, rs, x0, y0, z0, vx0, vy0, vz0, time, alpha = params
    vx0 *= KMS_TO_KPCGYR
    vy0 *= KMS_TO_KPCGYR
    vz0 *= KMS_TO_KPCGYR

    backward_trajectory = integrate_satellite(x0, y0, z0, vx0, vy0, vz0, logM, Rs, q, dirx, diry, dirz, -time)

    forward_trajectory  = integrate_satellite(*backward_trajectory[-1, :], logM, Rs, q, dirx, diry, dirz, time*alpha)

    theta_sat_forward = jnp.arctan2(forward_trajectory[:, 1], forward_trajectory[:, 0])
    theta_sat_forward = jnp.where(theta_sat_forward < 0, theta_sat_forward + TWOPI, theta_sat_forward)
    theta_sat_forward = jax_unwrap(theta_sat_forward)

    hessians  = jax.vmap(NFWHessian, in_axes=(0, 0, 0, None, None, None, None, None, None)) \
                        (forward_trajectory[:, 0], forward_trajectory[:, 1], forward_trajectory[:, 2], logM, Rs, q, dirx, diry, dirz)
    rj, vj, R = get_rj_vj_R(hessians, forward_trajectory, 10 ** logm)
    ic_particle_spray = create_ic_particle_spray(forward_trajectory, rj, vj, R, tail, seed)

    index = jnp.repeat(jnp.arange(0, N_STEPS, 1), N_PARTICLES // N_STEPS)
    theta_stream , xv_stream = jax.vmap(integrate_stream_spray, in_axes=(0, 0, 0, 0, 0, 0, 0, None, None, None, None, None, None, None, None, None, None, None)) \
        (index, ic_particle_spray[:, 0], ic_particle_spray[:, 1], ic_particle_spray[:, 2], ic_particle_spray[:, 3], ic_particle_spray[:, 4], ic_particle_spray[:, 5],
        theta_sat_forward, forward_trajectory, logM, Rs, q, dirx, diry, dirz, logm, rs, time*alpha)

    xv_stream *= jnp.array([1, 1, 1, KPCGYR_TO_KMS, KPCGYR_TO_KMS, KPCGYR_TO_KMS])  # Convert velocities back to km/s
    forward_trajectory *= jnp.array([1, 1, 1, KPCGYR_TO_KMS, KPCGYR_TO_KMS, KPCGYR_TO_KMS])  # Convert velocities back to km/s
    return theta_stream, xv_stream, theta_sat_forward, forward_trajectory