import jax
import jax.numpy as jnp
from jax import grad, hessian, random
jax.config.update("jax_enable_x64", True)

from astropy import units as auni

from astropy.constants import G
G = G.to(auni.kpc/auni.Msun*auni.km**2/auni.s**2).value # kpc (km/s)^2/Msun

KPC_TO_KM    = jnp.array( (1 * auni.kpc/auni.km).to(auni.km/auni.km).value)
GYR_TO_S     = jnp.array( (1 * auni.Gyr/auni.s).to(auni.s/auni.s).value)
# N_particles must be even and divisible by N_steps
N_PARTICLES  = 10000 # Denis wants more particles 
N_STEPS      = 500 # Time resolution
N_BINS       = 36

# Precompute constants once
_v1 = jnp.array([0.0, 0.0, 1.0])
_I3 = jnp.eye(3)

@jax.jit
def get_mat(x, y, z):
    # Create a fixed-shape vector from inputs
    v2 = jnp.array([x, y, z])
    # Normalize v2 in one step
    v2 = v2 / (jnp.linalg.norm(v2) + 1e-8)

    # Compute the angle using a fused dot and clip operation
    angle = jnp.arccos(jnp.clip(jnp.dot(_v1, v2), -1.0, 1.0))

    # Compute normalized rotation axis
    v3 = jnp.cross(_v1, v2)
    v3 = v3 / (jnp.linalg.norm(v3) + 1e-8)

    # Build the skew-symmetric matrix K for Rodrigues' formula
    K = jnp.array([
        [0, -v3[2], v3[1]],
        [v3[2], 0, -v3[0]],
        [-v3[1], v3[0], 0]
    ])

    sin_angle = jnp.sin(angle)
    cos_angle = jnp.cos(angle)

    # Compute rotation matrix using Rodrigues' formula
    rot_mat = _I3 + sin_angle * K + (1 - cos_angle) * jnp.dot(K, K)
    return rot_mat


@jax.jit
def NFW_potential(x, y, z, logM, Rs, q, dirx, diry, dirz):
    # Stack coordinates to ensure fixed-shape inputs
    r_input = jnp.stack([x, y, z])

    # Compute rotation matrix (should be well-optimized already)
    rot_mat = get_mat(dirx, diry, dirz)

    # Rotate coordinates
    r_vect = jnp.dot(rot_mat, r_input)  # No dynamic tracing

    # Extract rotated components safely
    rx, ry, rz = r_vect[0], r_vect[1], r_vect[2]

    # Compute radius safely (avoid tracing issues)
    r = jnp.sqrt(rx**2 + ry**2 + (rz / q) ** 2 + 1e-8)

    # Compute mass and potential
    M = 10**logM
    Phi = -G * M / r * jnp.log(1 + r / Rs)  # km²/s²

    return Phi

@jax.jit
def scalar_NFW_acceleration(x, y, z, logM, Rs, q, dirx, diry, dirz):
    def potential_wrapper(x, y, z):
        return NFW_potential(x, y, z, logM, Rs, q, dirx, diry, dirz)

    dPhidx, dPhidy, dPhidz = grad(potential_wrapper, argnums=(0, 1, 2))(x, y, z)

    # Use jnp.stack() instead of .T
    acc = -jnp.stack([dPhidx, dPhidy, dPhidz], axis=0)  # km² / s² / kpc

    return acc * GYR_TO_S  # km² / s / Gyr / kpc

@jax.jit
def vector_NFW_acceleration(x, y, z, logM, Rs, q, dirx, diry, dirz):
    def potential_wrapper(x, y, z):
        return NFW_potential(x, y, z, logM, Rs, q, dirx, diry, dirz)

    grad_fn = jax.vmap(grad(potential_wrapper, argnums=(0, 1, 2)), in_axes=(0, 0, 0))

    dPhidx, dPhidy, dPhidz = grad_fn(x, y, z)

    # Use jnp.stack() instead of .T
    acc = -jnp.stack([dPhidx, dPhidy, dPhidz], axis=-1)  # Shape: (N, 3)

    return acc * GYR_TO_S  # km² / s / Gyr / kpc

@jax.jit
def scalar_NFW_Hessian(x, y, z, logM, Rs, q, dirx, diry, dirz):
    def potential_wrapper(x, y, z):
        return NFW_potential(x, y, z, logM, Rs, q, dirx, diry, dirz)

    hessian_matrix = hessian(potential_wrapper, argnums=(0, 1, 2))(x, y, z)  # Shape: (3, 3)

    return hessian_matrix  # km² / s / Gyr / kpc²

@jax.jit
def vector_NFW_Hessian(x, y, z, logM, Rs, q, dirx, diry, dirz):
    def potential_wrapper(x, y, z):
        return NFW_potential(x, y, z, logM, Rs, q, dirx, diry, dirz)

    hessian_fn = jax.vmap(hessian(potential_wrapper, argnums=(0, 1, 2)), in_axes=(0, 0, 0))

    hessian_tuple = hessian_fn(x, y, z)  # Shape: (N, 3, 3)

    # Convert tuple to a single (N, 3, 3) array
    hessian_matrix = jnp.asarray(hessian_tuple).transpose(2, 0, 1)  # Shape: (N, 3, 3)

    return hessian_matrix

@jax.jit
def Plummer_potential(x, y, z, logm, rs, x_origin=0, y_origin=0, z_origin=0):
    M = 10**logm  # Convert log mass to mass
    r2 = (x - x_origin) ** 2 + (y - y_origin) ** 2 + (z - z_origin) ** 2
    Phi = -G * M / jnp.sqrt(r2 + rs**2)

    return Phi  # km² / s²

@jax.jit
def scalar_Plummer_acceleration(x, y, z, logm, rs, x_origin=0, y_origin=0, z_origin=0):
    def potential_wrapper(x, y, z):
        return Plummer_potential(x, y, z, logm, rs, x_origin, y_origin, z_origin)

    dPhidx, dPhidy, dPhidz = grad(potential_wrapper, argnums=(0, 1, 2))(x, y, z)

    # Use jnp.stack instead of .T
    acc = -jnp.stack([dPhidx, dPhidy, dPhidz], axis=0)  # km² / s² / kpc

    return acc * GYR_TO_S  # km² / s / Gyr / kpc

@jax.jit
def vector_Plummer_acceleration(x, y, z, logm, rs, x_origin=0, y_origin=0, z_origin=0):
    def potential_wrapper(x, y, z):
        return Plummer_potential(x, y, z, logm, rs, x_origin, y_origin, z_origin)

    grad_fn = jax.vmap(grad(potential_wrapper, argnums=(0, 1, 2)), in_axes=(0, 0, 0))

    dPhidx, dPhidy, dPhidz = grad_fn(x, y, z)

    # Use jnp.stack instead of .T
    acc = -jnp.stack([dPhidx, dPhidy, dPhidz], axis=-1)  # Shape: (N, 3)

    return acc * GYR_TO_S  # km² / s / Gyr / kpc

@jax.jit
def leapfrog_orbit_step(state, dt, logM, Rs, q, dirx, diry, dirz):
    x, y, z, vx, vy, vz = state

    ax, ay, az = scalar_NFW_acceleration(x, y, z, logM, Rs, q, dirx, diry, dirz)

    vx_half = vx + 0.5 * dt * ax * KPC_TO_KM**-1
    vy_half = vy + 0.5 * dt * ay * KPC_TO_KM**-1
    vz_half = vz + 0.5 * dt * az * KPC_TO_KM**-1

    x_new = x + dt * vx_half * GYR_TO_S * KPC_TO_KM**-1
    y_new = y + dt * vy_half * GYR_TO_S * KPC_TO_KM**-1
    z_new = z + dt * vz_half * GYR_TO_S * KPC_TO_KM**-1

    ax_new, ay_new, az_new = scalar_NFW_acceleration(x_new, y_new, z_new, logM, Rs, q, dirx, diry, dirz)

    vx_new = vx_half + 0.5 * dt * ax_new * KPC_TO_KM**-1
    vy_new = vy_half + 0.5 * dt * ay_new * KPC_TO_KM**-1
    vz_new = vz_half + 0.5 * dt * az_new * KPC_TO_KM**-1

    return (x_new, y_new, z_new, vx_new, vy_new, vz_new)

@jax.jit
def backward_integrate_orbit_leapfrog(x0, y0, z0, vx0, vy0, vz0, logM, Rs, q, dirx, diry, dirz, time):
    state = (x0, y0, z0, vx0, vy0, vz0)
    dt    = time/N_STEPS

    # Ensure scalar inputs are JAX arrays
    logM, Rs, q = jnp.asarray(logM), jnp.asarray(Rs), jnp.asarray(q)
    dirx, diry, dirz = jnp.asarray(dirx), jnp.asarray(diry), jnp.asarray(dirz)

    # Step function for JAX scan
    def step_fn(state, _):
        new_state = leapfrog_orbit_step(state, -dt, logM, Rs, q, dirx, diry, dirz)
        return new_state, jnp.stack(new_state)  # Ensuring shape consistency

    # Run JAX optimized loop (reverse integration order)
    _, trajectory = jax.lax.scan(step_fn, state, None, length=N_STEPS - 1)#, unroll=True)

    # Ensure trajectory shape is (MAX_LENGHT-1, 6)
    trajectory = jnp.array(trajectory)  # Shape: (MAX_LENGHT-1, 6)

    # Correct concatenation
    trajectory = jnp.vstack([trajectory[::-1], jnp.array(state)[None, :]])  # Shape: (MAX_LENGHT, 6)

    # Compute time steps
    time_steps = -jnp.arange(N_STEPS) * dt

    return trajectory, time_steps

@jax.jit
def get_rj_vj_R(hessians, orbit_sat, mass_sat):
    N = orbit_sat.shape[0]
    x, y, z, vx, vy, vz = orbit_sat.T

    # Compute angular momentum L
    Lx = y * vz - z * vy
    Ly = z * vx - x * vz
    Lz = x * vy - y * vx
    r = jnp.sqrt(x**2 + y**2 + z**2 + 1e-8)  # Regularization to prevent NaN
    L = jnp.sqrt(Lx**2 + Ly**2 + Lz**2 + 1e-8)

    # Rotation matrix (transform from host to satellite frame)
    R = jnp.stack([
        jnp.stack([x / r, y / r, z / r], axis=-1),
        jnp.stack([
            (y / r) * (Lz / L) - (z / r) * (Ly / L),
            (z / r) * (Lx / L) - (x / r) * (Lz / L),
            (x / r) * (Ly / L) - (y / r) * (Lx / L)
        ], axis=-1),
        jnp.stack([Lx / L, Ly / L, Lz / L], axis=-1),
    ], axis=-2)  # Shape: (N, 3, 3)

    # Compute second derivative of potential
    d2Phi_dr2 = -(
        x**2 * hessians[:, 0, 0] + y**2 * hessians[:, 1, 1] + z**2 * hessians[:, 2, 2] +
        2 * x * y * hessians[:, 0, 1] + 2 * y * z * hessians[:, 1, 2] + 2 * z * x * hessians[:, 0, 2]
    ) / r**2 * KPC_TO_KM**-2 * GYR_TO_S**-1  # 1 / s²

    # Compute Jacobi radius and velocity offset
    Omega = L / r**2 * KPC_TO_KM**-1  # 1 / s
    rj = ((mass_sat * G / (Omega**2 - d2Phi_dr2)) * KPC_TO_KM**-2 + 1e-8) ** (1. / 3)  # kpc
    vj = Omega * rj * KPC_TO_KM

    return rj, vj, R

@jax.jit
def create_ic_particle_spray(orbit_sat, rj, vj, R, tail=0, seed=111):
    key=random.PRNGKey(seed)
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
    key, subkey_x, subkey_z, subkey_vy, subkey_vz = random.split(key, 5)
    rx = random.normal(subkey_x, shape=(N_PARTICLES//N_STEPS * N,)) * disp_x + mean_x
    rz = random.normal(subkey_z, shape=(N_PARTICLES//N_STEPS * N,)) * disp_z * rj
    rvy = (random.normal(subkey_vy, shape=(N_PARTICLES//N_STEPS * N,)) * disp_vy + mean_vy) * vj * rx
    rvz = random.normal(subkey_vz, shape=(N_PARTICLES//N_STEPS * N,)) * disp_vz * vj
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
def leapfrog_stream_step(state, dt, logM, Rs, q, dirx, diry, dirz, logm, rs):
    x, y, z, vx, vy, vz, xp, yp, zp, vxp, vyp, vzp = state

    # Update Satellite Position
    axp, ayp, azp = scalar_NFW_acceleration(xp, yp, zp, logM, Rs, q, dirx, diry, dirz)

    vxp_half = vxp + 0.5 * dt * axp * KPC_TO_KM**-1
    vyp_half = vyp + 0.5 * dt * ayp * KPC_TO_KM**-1
    vzp_half = vzp + 0.5 * dt * azp * KPC_TO_KM**-1

    xp_new = xp + dt * vxp_half * GYR_TO_S * KPC_TO_KM**-1
    yp_new = yp + dt * vyp_half * GYR_TO_S * KPC_TO_KM**-1
    zp_new = zp + dt * vzp_half * GYR_TO_S * KPC_TO_KM**-1

    axp_new, ayp_new, azp_new = scalar_NFW_acceleration(xp_new, yp_new, zp_new, logM, Rs, q, dirx, diry, dirz)

    vxp_new = vxp_half + 0.5 * dt * axp_new * KPC_TO_KM**-1
    vyp_new = vyp_half + 0.5 * dt * ayp_new * KPC_TO_KM**-1
    vzp_new = vzp_half + 0.5 * dt * azp_new * KPC_TO_KM**-1

    # Update Stream Position
    ax, ay, az = scalar_NFW_acceleration(x, y, z, logM, Rs, q, dirx, diry, dirz) +  \
                    scalar_Plummer_acceleration(x, y, z, logm, rs, x_origin=xp, y_origin=yp, z_origin=zp) # km2 / s / Gyr / kpc

    vx_half = vx + 0.5 * dt * ax * KPC_TO_KM**-1 # km / s
    vy_half = vy + 0.5 * dt * ay * KPC_TO_KM**-1
    vz_half = vz + 0.5 * dt * az * KPC_TO_KM**-1

    x_new = x + dt * vx_half * GYR_TO_S * KPC_TO_KM**-1 # kpc
    y_new = y + dt * vy_half * GYR_TO_S * KPC_TO_KM**-1
    z_new = z + dt * vz_half * GYR_TO_S * KPC_TO_KM**-1

    ax_new, ay_new, az_new = scalar_NFW_acceleration(x_new, y_new, z_new, logM, Rs, q, dirx, diry, dirz) +  \
                                scalar_Plummer_acceleration(x_new, y_new, z_new, logm, rs, x_origin=xp_new, y_origin=yp_new, z_origin=zp_new) # km2 / s / Gyr / kpc

    vx_new = vx_half + 0.5 * dt * ax_new * KPC_TO_KM**-1 # km / s
    vy_new = vy_half + 0.5 * dt * ay_new * KPC_TO_KM**-1
    vz_new = vz_half + 0.5 * dt * az_new * KPC_TO_KM**-1

    return (x_new, y_new, z_new, vx_new, vy_new, vz_new, xp_new, yp_new, zp_new, vxp_new, vyp_new, vzp_new)

@jax.jit
def unwrap_step(theta_t, theta_unwrapped_prev):
    # bring the previous unwrapped back into [0, 2π)
    theta_prev_raw = jnp.mod(theta_unwrapped_prev, 2 * jnp.pi)
    # raw increment
    dtheta = theta_t - theta_prev_raw
    # wrap into (–π, π]
    dtheta = (dtheta + jnp.pi) % (2 * jnp.pi) - jnp.pi
    # accumulate
    return theta_unwrapped_prev + dtheta

@jax.jit
def forward_integrate_orbit_leapfrog(x0, y0, z0, vx0, vy0, vz0, logM, Rs, q, dirx, diry, dirz, time):
    state = (x0, y0, z0, vx0, vy0, vz0)
    dt    = time/N_STEPS

    # Step function for JAX scan
    def step_fn(state, _):
        new_state = leapfrog_orbit_step(state, dt, logM, Rs, q, dirx, diry, dirz)
        return new_state, jnp.stack(new_state)  # Ensuring shape consistency

    # Run JAX optimized loop (reverse integration order)
    _, trajectory = jax.lax.scan(step_fn, state, None, length=N_STEPS - 1)#, unroll=True)

    # Ensure trajectory shape is (MAX_LENGHT-1, 6)
    trajectory = jnp.array(trajectory)  # Shape: (MAX_LENGHT-1, 6)

    # Correct concatenation
    trajectory = jnp.vstack([jnp.array(state)[None, :], trajectory])  # Shape: (MAX_LENGHT, 6)

    # Compute time steps
    time_steps = jnp.arange(N_STEPS) * dt

    return trajectory, time_steps

@jax.jit
def forward_integrate_stream_leapfrog(index, x0, y0, z0, vx0, vy0, vz0,
                                        xv_sat, logM, Rs, q,
                                        dirx, diry, dirz, logm, rs, time):
    # State is a flat tuple of six scalars.
    xp, yp, zp, vxp, vyp, vzp = xv_sat[index]

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
        final_conditions = leapfrog_stream_step(initial_conditions, dt_here,
                                            logM, Rs, q, dirx, diry, dirz, logm, rs)
        
        theta = jnp.arctan2(final_conditions[1], final_conditions[0])
        theta = jax.lax.cond(theta < 0, lambda x: x + 2 * jnp.pi, lambda x: x, theta)

        theta = unwrap_step(theta, theta0)

        new_state = (theta, *final_conditions)

        # The carry and output must have the same structure.
        return new_state, _ # jnp.stack(new_state)

    # Run integration over the satellite trajectory (using all but the last row).
    trajectory, _ = jax.lax.scan(step_fn, state, None, length=N_STEPS - 1) #, unroll=True)
    # 'trajectory' is a tuple of six arrays, each of shape (N_STEPS,).

    return jnp.array(trajectory)

@jax.jit
def generate_stream(ic_particle_spray, xv_sat, logM, Rs, q,
                    dirx, diry, dirz, logm, rs, time):
    # There are 16 parameters to forward_integrate_stream_leapfrog:
    # 6 come from ic_particle_spray (one per coordinate),
    # and the remaining 10 are shared (xv_sat, logM, Rs, q, dirx, diry, dirz, logm, rs, time).
    index = jnp.repeat(jnp.arange(0, N_STEPS, 1), N_PARTICLES// N_STEPS)  # Shape: (N_PARTICLES,)

    xv_stream = jax.vmap(
        forward_integrate_stream_leapfrog,
        in_axes=(0, 0, 0, 0, 0, 0, 0,  # map over each column of ic_particle_spray
                    None, None, None, None, None, None, None, None, None, None)  # shared arguments
    )(index,
        ic_particle_spray[:, 0],  # x0
        ic_particle_spray[:, 1],  # y0
        ic_particle_spray[:, 2],  # z0
        ic_particle_spray[:, 3],  # vx0
        ic_particle_spray[:, 4],  # vy0
        ic_particle_spray[:, 5],  # vz0
        xv_sat, # (xp, yp, zp, vxp, vyp, vzp)
        logM, Rs, q,
        dirx, diry, dirz, logm, rs, time)

    return xv_stream

@jax.jit
def jax_unwrap(theta):
    dtheta = jnp.diff(theta)
    dtheta_unwrapped = jnp.where(dtheta < -jnp.pi, dtheta + 2 * jnp.pi,
                         jnp.where(dtheta > jnp.pi, dtheta - 2 * jnp.pi, dtheta))
    return jnp.concatenate([theta[:1], theta[:1] + jnp.cumsum(dtheta_unwrapped)])

@jax.jit
def bin_stream(theta_stream, r_stream, x_stream, y_stream, vz_stream, min_count):
    # Step 1: Create bin edges and assign particles to bins
    bin_edges   = jnp.linspace(-2 * jnp.pi, 2 * jnp.pi, N_BINS + 1)
    bin_indices = jnp.digitize(theta_stream, bin_edges, right=True)

    # Step 2: Per-bin median computation
    def per_bin_median(bin_idx, bin_ids, r, x, y, vz):
        mask = bin_ids == bin_idx
        count = jnp.sum(mask).astype(jnp.float64)

        def compute_medians():
            return (
                count,
                jnp.nanmean(jnp.where(mask, r, jnp.nan)),
                jnp.nanstd(jnp.where(mask, r, jnp.nan)),
                jnp.nanmedian(jnp.where(mask, x, jnp.nan)),
                jnp.nanmedian(jnp.where(mask, y, jnp.nan)),
                jnp.nanmedian(jnp.where(mask, vz, jnp.nan))
            )

        return jax.lax.cond(count > min_count, compute_medians, lambda: (jnp.nan, jnp.nan, jnp.nan, jnp.nan, jnp.nan, jnp.nan))

    # Step 3: Vectorize
    all_bins = jnp.arange(1, N_BINS + 1)
    count, r_meds, w_meds, x_meds, y_meds, vz_meds = jax.vmap(per_bin_median, in_axes=(0, None, None, None, None, None))(
        all_bins, bin_indices, r_stream, x_stream, y_stream, vz_stream
    )

    return count, r_meds, w_meds, x_meds, y_meds, vz_meds

@jax.jit
def jax_stream_model(logM, Rs, q, dirx, diry, dirz, logm, rs,
                        x0, y0, z0, vx0, vy0, vz0, time, alpha, tail, min_count, seed=111):
    # Condition: check that all differences in theta_sat are positive.
    # This ensures that the satellite is moving in a consistent direction.
    xv_sat, _ = backward_integrate_orbit_leapfrog(x0, y0, z0, vx0, vy0, vz0,
                                                    logM, Rs, q, dirx, diry, dirz,
                                                    time)
    theta_sat = jnp.arctan2(xv_sat[:, 1], xv_sat[:, 0])
    theta_sat = jnp.where(theta_sat < 0, theta_sat + 2 * jnp.pi, theta_sat)
    theta_sat = jax_unwrap(theta_sat)

    cond = jnp.all(jnp.diff(theta_sat) > 0)

    # # Define the branch that computes the stream.
    def true_branch(_):
        xv_sat, _ = backward_integrate_orbit_leapfrog(x0, y0, z0, vx0, vy0, vz0,
                                                    logM, Rs, q, dirx, diry, dirz,
                                                    time)
        # theta_sat = jnp.arctan2(xv_sat[:, 1], xv_sat[:, 0])
        # theta_sat = jnp.where(theta_sat < 0, theta_sat + 2 * jnp.pi, theta_sat)
        # theta_sat = jax_unwrap(theta_sat)

        xv_sat_forward, _ = forward_integrate_orbit_leapfrog(xv_sat[0, 0], xv_sat[0, 1], xv_sat[0, 2], xv_sat[0,3], xv_sat[0, 4], xv_sat[0, 5],
                                                    logM, Rs, q, dirx, diry, dirz,
                                                    time*alpha)        
        theta_sat_forward = jnp.arctan2(xv_sat_forward[:, 1], xv_sat_forward[:, 0])
        theta_sat_forward = jnp.where(theta_sat_forward < 0, theta_sat_forward + 2 * jnp.pi, theta_sat_forward)
        theta_sat_forward = jax_unwrap(theta_sat_forward)

        hessians = vector_NFW_Hessian(xv_sat_forward[:, 0], xv_sat_forward[:, 1], xv_sat_forward[:, 2],
                                        logM, Rs, q, dirx, diry, dirz)
        rj, vj, R = get_rj_vj_R(hessians, xv_sat_forward, 10 ** logm)
        ic_particle_spray = create_ic_particle_spray(xv_sat_forward, rj, vj, R, tail, seed=seed)

        xv_stream = generate_stream(ic_particle_spray, xv_sat_forward, logM, Rs, q,
                                    dirx, diry, dirz, logm, rs, time)
        
        # === Process angles as a function of Progenitor ===
        # Count how many complete 2pi rotations have been accumulated (integer division).
        theta_count = jnp.floor_divide(theta_sat_forward, 2 * jnp.pi)

        final_theta_stream = (
            xv_stream[:, 0] #jnp.sum(theta_stream * diagonal_matrix, axis=1)
            - theta_sat_forward[-1]
            + jnp.repeat(theta_count,  N_PARTICLES// N_STEPS) * 2 * jnp.pi
            )

        algin_reference = theta_sat_forward[-1]- theta_count[-1]*(2*jnp.pi) # Make sure the angle of reference is at theta=0

        final_theta_stream += (1 - jnp.sign(algin_reference - jnp.pi))/2 * algin_reference + \
                                (1 + jnp.sign(algin_reference - jnp.pi))/2 * (algin_reference - 2 * jnp.pi)
        
        # x_stream, y_stream, theta_stream, vz_stream = \
        #     get_stream_and_unwrap_theta(xv_stream, xv_sat_forward)

        # Remove the last 1% of particules
        x_stream     = xv_stream[:-(N_PARTICLES//100), 1]
        y_stream     = xv_stream[:-(N_PARTICLES//100), 2]
        theta_stream = final_theta_stream[:-(N_PARTICLES//100)]
        vz_stream    = xv_stream[:-(N_PARTICLES//100), -1]

        r_stream = jnp.sqrt(x_stream**2 + y_stream**2)
        count, r_meds, w_meds, x_meds, y_meds, vz_meds = \
            bin_stream(theta_stream, r_stream, x_stream, y_stream, vz_stream,
                        min_count=min_count)

        return theta_stream, x_stream, y_stream, vz_stream, \
                count, r_meds, w_meds, x_meds, y_meds, vz_meds

    # Define the branch to use if condition is false.
    # Here we return dummy arrays with the same shapes and dtypes as in the true branch.
    # (In your actual use, you might choose to return a special flag value.)
    def false_branch(_):
            # Return dummy arrays with identical shapes and dtypes as in the true branch.
            # Adjust these shapes to match your expected outputs.
            # For example, here we assume:
            #   - The stream outputs have shape (1000,) (first 4 outputs).
            #   - The binned outputs have shape (36,) (last 4 outputs).
            # dummy_theta = jnp.full((99*N_PARTICLES//100,), jnp.nan, dtype=jnp.float32)
            # dummy_x     = jnp.full((99*N_PARTICLES//100,), jnp.nan, dtype=jnp.float32)
            # dummy_y     = jnp.full((99*N_PARTICLES//100,), jnp.nan, dtype=jnp.float32)
            # dummy_vz    = jnp.full((99*N_PARTICLES//100,), jnp.nan, dtype=jnp.float32)
            # dummy_r_meds = jnp.full((N_BINS,), jnp.nan, dtype=jnp.float32)
            # dummy_w_meds = jnp.full((N_BINS,), jnp.nan, dtype=jnp.float32)
            # dummy_x_meds = jnp.full((N_BINS,), jnp.nan, dtype=jnp.float32)
            # dummy_y_meds = jnp.full((N_BINS,), jnp.nan, dtype=jnp.float32)
            # dummy_vz_meds = jnp.full((N_BINS,), jnp.nan, dtype=jnp.float32)

            dummy_theta = jnp.full((99*N_PARTICLES//100,), jnp.nan, dtype=jnp.float64)
            dummy_x     = jnp.full((99*N_PARTICLES//100,), jnp.nan, dtype=jnp.float64)
            dummy_y     = jnp.full((99*N_PARTICLES//100,), jnp.nan, dtype=jnp.float64)
            dummy_vz    = jnp.full((99*N_PARTICLES//100,), jnp.nan, dtype=jnp.float64)
            dummy_count = jnp.full((N_BINS,), jnp.nan, dtype=jnp.float64)
            dummy_r_meds = jnp.full((N_BINS,), jnp.nan, dtype=jnp.float64)
            dummy_w_meds = jnp.full((N_BINS,), jnp.nan, dtype=jnp.float64)
            dummy_x_meds = jnp.full((N_BINS,), jnp.nan, dtype=jnp.float64)
            dummy_y_meds = jnp.full((N_BINS,), jnp.nan, dtype=jnp.float64)
            dummy_vz_meds = jnp.full((N_BINS,), jnp.nan, dtype=jnp.float64)

            return dummy_theta, dummy_x, dummy_y, dummy_vz, dummy_count, dummy_r_meds, dummy_w_meds, dummy_x_meds, dummy_y_meds, dummy_vz_meds

    # Use lax.cond to select the branch.
    return jax.lax.cond(cond, true_branch, false_branch, operand=None)

@jax.jit
def jax_stream_orbit(logM, Rs, q, dirx, diry, dirz,
                        x0, y0, z0, vx0, vy0, vz0, time, alpha, min_count=0):

    xv_sat, _ = forward_integrate_orbit_leapfrog(x0, y0, z0, vx0, vy0, vz0,
                                            logM, Rs, q, dirx, diry, dirz,
                                            time*alpha)
    x_sat = xv_sat[:, 0]
    y_sat = xv_sat[:, 1]
    vz_sat = xv_sat[:, -1]
    r_sat = jnp.sqrt(x_sat**2+y_sat**2)

    theta_sat = jnp.arctan2(y_sat, x_sat)
    theta_sat = jnp.where(theta_sat < 0, theta_sat + 2 * jnp.pi, theta_sat)
    theta_sat = jax_unwrap(theta_sat)

    r_meds, w_meds, x_meds, y_meds, vz_meds = bin_stream(theta_sat, r_sat, x_sat, y_sat, vz_sat, min_count=min_count)

    return theta_sat, x_sat, y_sat, vz_sat, \
            r_meds, w_meds, x_meds, y_meds, vz_meds

@jax.jit
def sample_params_data(q_true, seed):
    # seed = np.random.randint(0, 2**32 - 1)  # Ensure it's within JAX's valid range

    key = random.PRNGKey(seed)  # Set seed for reproducibility
    # Split key once for all parameters
    keys = random.split(key, 9)  # Generate enough subkeys at once

    # Generate random variables
    logM = random.uniform(keys[0], shape=(), minval=11, maxval=14)
    Rs   = random.uniform(keys[1], shape=(), minval=5, maxval=25)
    q    = q_true #random.uniform(keys[2], shape=(), minval=0.5, maxval=1.5)
    dirx, diry, dirz = random.normal(keys[7], shape=(3,))    # Mean 0, Std 1
    dirz = jnp.abs(dirz)  # Ensure positive direction

    logm = random.uniform(keys[3], shape=(), minval=7, maxval=9)
    rs   = random.uniform(keys[4], shape=(), minval=1, maxval=3)

    # Generate normal-distributed variables
    x0, z0 = random.normal(keys[5], shape=(2,)) * 150     # Mean 0, Std 50
    x0 = jnp.abs(x0)  # Ensure positive position
    z0 = jnp.abs(z0)
    y0 = 0. # Set to 0

    vx0, vy0, vz0 = random.normal(keys[6], shape=(3,)) * 250  # Mean 0, Std 50
    vy0 = jnp.abs(vy0)

    # Generate time
    time = random.uniform(keys[8], shape=(), minval=1, maxval=4)
    alpha = 1.

    return logM, Rs, q, dirx, diry, dirz, logm, rs, x0, y0, z0, vx0, vy0, vz0, time, alpha