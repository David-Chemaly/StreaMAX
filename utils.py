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

        return count, jnp.nanmedian(r_in_bin), (jnp.nanpercentile(r_in_bin, 84) - jnp.nanpercentile(r_in_bin, 16)) / 2
    # Step 3: Vectorize
    all_bins = jnp.arange(1, n_bins + 1)
    count, r_bin, w_bin = jax.vmap(per_bin_median, in_axes=(0, None, None))(all_bins, bin_indices, r_stream)

    return count, theta_bin, r_bin, w_bin

@jax.jit
def get_track_from_data(theta_stream, x_stream, y_stream, theta_data):
    r_stream = jnp.sqrt(x_stream**2 + y_stream**2)
    delta = jnp.diff(theta_data).mean()/2
    lefts  = theta_data - delta
    rights =  theta_data + delta
    inside = (theta_stream[:, None] >= lefts[None, :]) & (theta_stream[:, None] < rights[None, :])
    has_bin = inside.any(axis=1)
    idx = inside.argmax(axis=1)
    bin_indices = jnp.where(has_bin, idx, -1)

    # Step 2: Per-bin median computation
    def per_bin_median(bin_idx, bin_ids, r):
        mask     = bin_ids == bin_idx
        count    = jnp.sum(mask)
        r_in_bin = jnp.where(mask, r, jnp.nan)

        return count, jnp.nanmedian(r_in_bin), (jnp.nanpercentile(r_in_bin, 84) - jnp.nanpercentile(r_in_bin, 16)) / 2
    # Step 3: Vectorize
    all_bins = jnp.arange(0, len(theta_data))
    count, r_bin, w_bin = jax.vmap(per_bin_median, in_axes=(0, None, None))(all_bins, bin_indices, r_stream)

    return count, r_bin, w_bin

@jax.jit
def get_track_weights(theta_stream, x_stream, y_stream, weights, n_bins=36):
    # Step 1: Create bin edges and assign particles to bins
    r_stream = jnp.sqrt(x_stream**2 + y_stream**2)
    bin_edges   = jnp.linspace(-2 * jnp.pi, 2 * jnp.pi, n_bins + 1)
    theta_bin   = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = jnp.digitize(theta_stream, bin_edges, right=True)

    # Step 2: Per-bin median computation
    def per_bin_median(bin_idx, bin_ids, r, weights):
        mask     = bin_ids == bin_idx
        count    = jnp.sum(mask*weights)
        r_in_bin = jnp.where(mask, r, jnp.nan)
        weights_in_bin = jnp.where(mask, weights, jnp.nan)
        weights_in_bin /= jnp.nansum(weights_in_bin)

        mean = jnp.nansum(weights_in_bin*r_in_bin)
        return count, mean, jnp.sqrt(jnp.nansum(weights_in_bin*(r_in_bin - mean)**2))

    # Step 3: Vectorize
    all_bins = jnp.arange(1, n_bins + 1)
    count, r_bin, w_bin = jax.vmap(per_bin_median, in_axes=(0, None, None, None))(all_bins, bin_indices, r_stream, weights)

    return count, theta_bin, r_bin, w_bin

@jax.jit
def get_q(dirx, diry, dirz, q_min=0.5, q_max=2.0):
    """
    Computes the axis ratio q from the direction vector components. Uniform [0.5, 1.5].
    """
    r  = jnp.sqrt(dirx**2 + diry**2 + dirz**2) 
    q  = jnp.exp(-r**2/2) * (jnp.sqrt(jnp.pi) * jnp.exp(r**2/2) * jax.scipy.special.erf(r/jnp.sqrt(2)) - jnp.sqrt(2)*r)/jnp.sqrt(jnp.pi)
    q  = (q_max-q_min)*q + q_min

    return q

@jax.jit
def get_q_proj(q, z):
    return jnp.sqrt(q**2 + (1-q**2)*z**2)

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
def inference_first(theta_stream, xv_stream, refs, weights_refs, S, seed=111, disp_x=0.5, disp_v=0.5):
    key=jax.random.PRNGKey(seed)
    disp = jnp.array([disp_x, disp_x, disp_x, disp_v, disp_v, disp_v])

    samples = jax.random.normal(key, shape=(10, len(refs), 6)) * disp + refs
    weights_samples = jnp.repeat(weights_refs, 10)
    samples_final = xv_stream[:, :3] + jnp.einsum('ijk, nik -> nij', S, samples - refs)

    theta_samples = jnp.arctan2(samples_final[:, :,1], samples_final[:, :, 0])
    theta_samples = jnp.where(theta_samples < 0, theta_samples + 2 * jnp.pi, theta_samples)
    unwrapped_theta_samples = jax.vmap(unwrap_step, in_axes=(0, None))(theta_samples, theta_stream)

    return unwrapped_theta_samples.reshape(-1), samples_final.reshape(-1, 3), weights_samples.reshape(-1)

@jax.jit
def inference_second(theta_stream, xv_stream, refs, S, T, seed=111, disp_x=0.5, disp_v=0.5):
    key=jax.random.PRNGKey(seed)
    disp = jnp.array([disp_x, disp_x, disp_x, disp_v, disp_v, disp_v])

    samples = jax.random.normal(key, shape=(10, len(refs), 6)) * disp + refs
    samples_final = xv_stream[:, :3] + jnp.einsum('ijk, nik -> nij', S, samples - refs) \
                                        + 0.5*jnp.einsum('ijkl, nik, nil -> nij', T, samples - refs, samples - refs)

    theta_samples = jnp.arctan2(samples_final[:, :,1], samples_final[:, :, 0])
    theta_samples = jnp.where(theta_samples < 0, theta_samples + 2 * jnp.pi, theta_samples)
    unwrapped_theta_samples = jax.vmap(unwrap_step, in_axes=(0, None))(theta_samples, theta_stream)

    return unwrapped_theta_samples.reshape(-1), samples_final.reshape(-1, 3)

### For Real Image ###

from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo
import numpy as np

def get_residuals_and_mask(path, sga, name, vminperc=35, vmaxperc=90):
    # Load Residuals
    with fits.open(f"{path}/{name}/data.fits") as hdul:
        header = hdul[0].header
        data = hdul[0].data
    with fits.open(f"{path}/{name}/model.fits") as hdul:
        model = hdul[0].data
    residual = data - model
    residual = np.median(residual, axis=0)
    mm = np.nanpercentile(residual, [vminperc, vmaxperc])
    residual = np.nan_to_num(np.clip(residual, mm[0], mm[1]), 0.0)

    # Load Mask
    with fits.open(f"{path}/{name}/mask.fits") as hdul:
        mask = hdul[0].data
    mask = mask/mask.max() # This assumes that only one mask is present

    # Get Redshift and pixel scale
    sga_name = name.split('_')[0]
    PA = sga[sga['GALAXY'] == sga_name]['PA'].data[0]
    z_redshift = sga[sga['GALAXY'] == sga_name]['Z_LEDA'].data[0]
    pixel_to_deg = abs(header['PC1_1'])
    pixel_to_kpc = pixel_to_deg * np.pi / 180 * cosmo.comoving_transverse_distance(z_redshift).value * 1000

    return residual, mask, z_redshift, pixel_to_kpc, PA

import math
def halo_mass_from_stellar_mass(M_star, 
                                N=0.0351, log10_M1=11.59, beta=1.376, gamma=0.608,
                                mmin=1e9, mmax=3e16, tol=1e-6, max_iter=200):
    """
    Return halo mass M_h [Msun] for a given stellar mass M_star [Msun]
    using the Moster+2013 z=0 SHMR (median relation).
    """
    def mstar_from_mh(Mh):
        x = Mh / (10**log10_M1)
        return 2*N*Mh / (x**(-beta) + x**gamma)

    a, b = mmin, mmax
    for _ in range(max_iter):
        mid = 10**((math.log10(a)+math.log10(b))/2)
        if mstar_from_mh(mid) > M_star:
            b = mid
        else:
            a = mid
        if abs(math.log10(b) - math.log10(a)) < tol:
            return 10**((math.log10(a)+math.log10(b))/2)
    return 10**((math.log10(a)+math.log10(b))/2)