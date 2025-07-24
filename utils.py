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

@functools.partial(jax.jit, static_argnums=(-2, -1, ))
def create_ic_particle_spray(orbit_sat, rj, vj, R, tail=0, seed=111, N_PARTICLES=10000, N_STEPS=100):
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
def jax_unwrap(theta):
    dtheta = jnp.diff(theta)
    dtheta_unwrapped = jnp.where(dtheta < -jnp.pi, dtheta + 2 * jnp.pi,
                         jnp.where(dtheta > jnp.pi, dtheta - 2 * jnp.pi, dtheta))
    return jnp.concatenate([theta[:1], theta[:1] + jnp.cumsum(dtheta_unwrapped)])
