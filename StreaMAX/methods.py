import jax
import jax.numpy as jnp
from functools import partial

from .constants import G, DEG_TO_RAD

# ---------- Fardal et al. 2015 ----------
@partial(jax.jit, static_argnames=('n_particles', 'n_steps'))
def create_ic_particle_spray_Fardal2015(orbit_sat, rj, vj, R, n_particles, n_steps, tail=0, seed=111):
    key=jax.random.PRNGKey(seed)
    N = rj.shape[0]

    tile = jax.lax.cond(tail == 0, lambda _: jnp.tile(jnp.array([1, -1]), n_particles//2),
                        lambda _: jax.lax.cond(tail == 1, lambda _: jnp.tile(jnp.array([-1, -1]), n_particles//2),
                        lambda _: jnp.tile(jnp.array([1, 1]), n_particles//2), None), None)

    rj = jnp.repeat(rj, n_particles//n_steps) * tile
    vj = jnp.repeat(vj, n_particles//n_steps) * tile
    R  = jnp.repeat(R, n_particles//n_steps, axis=0)  # Shape: (2N, 3, 3)

    # Parameters for position and velocity offsets
    mean_x, disp_x = 2.0, 0.5
    disp_z = 0.5
    mean_vy, disp_vy = 0.3, 0.5
    disp_vz = 0.5

    # Generate random samples for position and velocity offsets
    key, subkey_x, subkey_z, subkey_vy, subkey_vz = jax.random.split(key, 5)
    rx = jax.random.normal(subkey_x, shape=(n_particles//n_steps * N,)) * disp_x + mean_x
    rz = jax.random.normal(subkey_z, shape=(n_particles//n_steps * N,)) * disp_z * rj
    rvy = (jax.random.normal(subkey_vy, shape=(n_particles//n_steps * N,)) * disp_vy + mean_vy) * vj * rx
    rvz = jax.random.normal(subkey_vz, shape=(n_particles//n_steps * N,)) * disp_vz * vj
    rx *= rj  # Scale x displacement by rj

    # Position and velocity offsets in the satellite reference frame
    offset_pos = jnp.column_stack([rx, jnp.zeros_like(rx), rz])  # Shape: (2N, 3)
    offset_vel = jnp.column_stack([jnp.zeros_like(rx), rvy, rvz])  # Shape: (2N, 3)

    # Transform to the host-centered frame
    orbit_sat_repeated = jnp.repeat(orbit_sat, n_particles//n_steps, axis=0)  # More efficient than tile+reshape
    offset_pos_transformed = jnp.einsum('ni,nij->nj', offset_pos, R)
    offset_vel_transformed = jnp.einsum('ni,nij->nj', offset_vel, R)

    ic_stream = orbit_sat_repeated + jnp.concatenate([offset_pos_transformed, offset_vel_transformed], axis=-1)

    return ic_stream  # Shape: (N_particule, 6)


# ---------- Chen et al. 2025 ----------
@partial(jax.jit, static_argnames=('n_particles', 'n_steps'))
def create_ic_particle_spray_Chen2025(orbit_sat, rj, vj, R, log_mass_sat, n_particles, n_steps, tail=0, seed=111):
    '''
    https://doi.org/10.3847/1538-4365/ad9904
    '''

    key=jax.random.PRNGKey(seed)
    N = rj.shape[0]

    tile = jax.lax.cond(tail == 0, lambda _: jnp.tile(jnp.array([1, -1]), n_particles//2),
                        lambda _: jax.lax.cond(tail == 1, lambda _: jnp.tile(jnp.array([-1, -1]), n_particles//2),
                        lambda _: jnp.tile(jnp.array([1, 1]), n_particles//2), None), None)

    rj = jnp.repeat(rj, n_particles//n_steps)
    vj = jnp.repeat(vj, n_particles//n_steps)
    R  = jnp.repeat(R, n_particles//n_steps, axis=0)  # Shape: (2N, 3, 3)
    log_mass_sat = jnp.repeat(log_mass_sat, n_particles//n_steps)

    # Parameters for position and velocity offsets
    mean = jnp.array([1.6, -30, 0, 1, 20, 0])
    cov = jnp.array(
        [
            [0.1225, 0, 0, 0, -4.9, 0],
            [0, 529, 0, 0, 0, 0],
            [0, 0, 144, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [-4.9, 0, 0, 0, 400, 0],
            [0, 0, 0, 0, 0, 484],
        ]
    )

    posvel = jax.random.multivariate_normal(key, mean, cov, shape=(n_particles//n_steps * N,), method='svd')

    Dr = posvel[:, 0] * rj

    v_esc = jnp.sqrt(2 * G * 10**log_mass_sat / Dr)
    Dv = posvel[:, 3] * v_esc

    # convert degrees to radians
    phi = posvel[:, 1] * DEG_TO_RAD
    theta = posvel[:, 2] * DEG_TO_RAD
    alpha = posvel[:, 4] * DEG_TO_RAD
    beta = posvel[:, 5] * DEG_TO_RAD

    ctheta, stheta = jnp.cos(theta), jnp.sin(theta)
    cphi, sphi = jnp.cos(phi), jnp.sin(phi)
    calpha, salpha = jnp.cos(alpha), jnp.sin(alpha)
    cbeta, sbeta = jnp.cos(beta), jnp.sin(beta)

    rx = (Dr * ctheta * cphi) * tile
    ry = (Dr * ctheta * sphi) * tile
    rz = (Dr * stheta)

    rvx = (Dv * cbeta * calpha) * tile
    rvy = (Dv * cbeta * salpha) * tile
    rvz = (Dv * sbeta)

    # Position and velocity offsets in the satellite reference frame
    offset_pos = jnp.column_stack([rx, ry, rz])  # Shape: (2N, 3)
    offset_vel = jnp.column_stack([rvx, rvy, rvz])  # Shape: (2N, 3)

    # Transform to the host-centered frame
    orbit_sat_repeated = jnp.repeat(orbit_sat, n_particles//n_steps, axis=0)  # More efficient than tile+reshape
    offset_pos_transformed = jnp.einsum('ni,nij->nj', offset_pos, R)
    offset_vel_transformed = jnp.einsum('ni,nij->nj', offset_vel, R)

    ic_stream = orbit_sat_repeated + jnp.concatenate([offset_pos_transformed, offset_vel_transformed], axis=-1)

    return ic_stream  # Shape: (N_particule, 6)