import functools
import jax
import jax.numpy as jnp

from constants import GYR_TO_S, KPC_TO_KM, EPSILON, G

@jax.jit
def get_mat(x, y, z):
    v1 = jnp.array([0.0, 0.0, 1.0])
    I3 = jnp.eye(3)

    # Create a fixed-shape vector from inputs
    v2 = jnp.array([x, y, z])
    # Normalize v2 in one step
    v2 = v2 / (jnp.linalg.norm(v2) + EPSILON)

    # Compute the angle using a fused dot and clip operation
    angle = jnp.arccos(jnp.clip(jnp.dot(v1, v2), -1.0, 1.0))

    # Compute normalized rotation axis
    v3 = jnp.cross(v1, v2)
    v3 = v3 / (jnp.linalg.norm(v3) + EPSILON)

    # Build the skew-symmetric matrix K for Rodrigues' formula
    K = jnp.array([
        [0, -v3[2], v3[1]],
        [v3[2], 0, -v3[0]],
        [-v3[1], v3[0], 0]
    ])

    sin_angle = jnp.sin(angle)
    cos_angle = jnp.cos(angle)

    # Compute rotation matrix using Rodrigues' formula
    rot_mat = I3 + sin_angle * K + (1 - cos_angle) * jnp.dot(K, K)
    return rot_mat

@jax.jit
def get_rj_vj_R(hessians, orbit_sat, mass_sat):
    x, y, z, vx, vy, vz = orbit_sat.T

    # Compute angular momentum L
    Lx = y * vz - z * vy
    Ly = z * vx - x * vz
    Lz = x * vy - y * vx
    r = jnp.sqrt(x**2 + y**2 + z**2)  # Regularization to prevent NaN
    L = jnp.sqrt(Lx**2 + Ly**2 + Lz**2)

    # Rotation matrix (transform from host to satellite frame)
    R = jnp.stack([
        jnp.stack([x / r, y / r, z / r], axis=-1),
        -jnp.stack([
            (y / r) * (Lz / L) - (z / r) * (Ly / L),
            (z / r) * (Lx / L) - (x / r) * (Lz / L),
            (x / r) * (Ly / L) - (y / r) * (Lx / L)
        ], axis=-1),
        jnp.stack([Lx / L, Ly / L, Lz / L], axis=-1),
    ], axis=-2)  # Shape: (N, 3, 3)

    # Compute second derivative of potential
    d2Phi_dr2 = (
        x**2 * hessians[:, 0, 0] + y**2 * hessians[:, 1, 1] + z**2 * hessians[:, 2, 2] +
        2 * x * y * hessians[:, 0, 1] + 2 * y * z * hessians[:, 1, 2] + 2 * z * x * hessians[:, 0, 2]
    ) / r**2 # 1 / Gyr²

    # Compute Jacobi radius and velocity offset
    Omega = L / r**2  # 1 / Gyr
    rj = ((mass_sat * G / (Omega**2 - d2Phi_dr2))) ** (1. / 3)  # kpc
    vj = Omega * rj

    return rj, vj, R

@jax.jit
def jax_unwrap(theta):
    dtheta = jnp.diff(theta)
    dtheta_unwrapped = jnp.where(dtheta < -jnp.pi, dtheta + 2 * jnp.pi,
                         jnp.where(dtheta > jnp.pi, dtheta - 2 * jnp.pi, dtheta))
    return jnp.concatenate([theta[:1], theta[:1] + jnp.cumsum(dtheta_unwrapped)])

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
def get_track(theta_stream, x_stream, y_stream, n_bins=36):
    # Step 1: Create bin edges and assign particles to bins
    r_stream = jnp.sqrt(x_stream**2 + y_stream**2)
    bin_edges   = jnp.linspace(-2 * jnp.pi, 2 * jnp.pi, n_bins + 1)
    theta_bin   = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = jnp.digitize(theta_stream, bin_edges, right=True)

    # Step 2: Per-bin median computation
    def per_bin_median(bin_idx, bin_ids, r):
        mask     = bin_ids == bin_idx
        count    = jnp.sum(mask)
        r_in_bin = jnp.where(mask, r, jnp.nan)

        return count, jnp.nanmedian(r_in_bin), (jnp.nanpercentile(r_in_bin, 84) - jnp.nanpercentile(r_in_bin, 16))/2

    # Step 3: Vectorize
    all_bins = jnp.arange(1, n_bins + 1)
    count, r_bin, w_bin = jax.vmap(per_bin_median, in_axes=(0, None, None))(all_bins, bin_indices, r_stream)

    return count, theta_bin, r_bin, w_bin

@jax.jit
def get_q(dirx, diry, dirz):
    """
    Computes the axis ratio q from the direction vector components. Uniform [0.5, 1.5].
    """
    r  = jnp.sqrt(dirx**2 + diry**2 + dirz**2) 
    q  = jnp.exp(-r**2/2) * (jnp.sqrt(jnp.pi) * jnp.exp(r**2/2) * jax.scipy.special.erf(r/jnp.sqrt(2)) - jnp.sqrt(2)*r)/jnp.sqrt(jnp.pi)
    q += 0.5

    return q

@jax.jit
def unwrap_stream_from_unwrapped_orbit(theta_sat, theta_stream, n_particles=10000, n_steps=500):
    theta_count = jnp.floor_divide(theta_sat, 2 * jnp.pi)

    final_theta_stream = (
        theta_stream #jnp.sum(theta_stream * diagonal_matrix, axis=1)
        - theta_sat[-1]
    + jnp.repeat(theta_count,  n_particles// n_steps) * 2 * jnp.pi
    )

    algin_reference = theta_sat[-1]- theta_count[-1]*(2*jnp.pi) # Make sure the angle of reference is at theta=0

    final_theta_stream += (1 - jnp.sign(algin_reference - jnp.pi))/2 * algin_reference + \
                            (1 + jnp.sign(algin_reference - jnp.pi))/2 * (algin_reference - 2 * jnp.pi)
    
    return final_theta_stream

@jax.jit
def find_percentile_arg(nb, vt, mask, gap):
    arg_keep = jnp.nanargmin(jnp.abs(vt*mask - jnp.nanpercentile(vt*mask, 100 * gap * (nb + 1))))
    return arg_keep

@functools.partial(jax.jit, static_argnums=(-1,))
def percentile_block(count_stream, theta_stream, xv_stream, vt, mask, arg_remove, nb_out, num_bin):
    gap = 1/(num_bin + 1) 
    arg_keep = jax.vmap(find_percentile_arg, in_axes=(0, None, None, None))(jnp.arange(num_bin), vt, mask, gap)
    arg_remove   = arg_remove.at[arg_keep].set(1.)
    count_stream = count_stream.at[arg_keep].set(nb_out / num_bin)

    return count_stream * arg_remove, theta_stream * arg_remove, xv_stream * arg_remove[:, None]

@functools.partial(jax.jit, static_argnums=(-1,))
def update_streams(count_stream, theta_stream, xv_stream, vt, mask, arg_remove, nb_out, bool_mask, num_bin):
    def then_fn(_):
        return count_stream + bool_mask, theta_stream, xv_stream

    def else_fn(_):
        return percentile_block(count_stream, theta_stream, xv_stream, vt, mask, arg_remove, nb_out, num_bin)

    return jax.lax.cond(nb_out <= num_bin, then_fn, else_fn, operand=None)