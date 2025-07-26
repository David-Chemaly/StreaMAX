import jax
import jax.numpy as jnp
import functools

from integrants import integrate_satellite, integrate_stream_spray
from utils import get_rj_vj_R, jax_unwrap, create_ic_particle_spray
from potentials import NFWHessian


@functools.partial(jax.jit, static_argnums=(2, 3,))
def generate_stream_spray(params,  seed, n_steps=500, n_particles=10000, tail=0):
    """
    Generates a stream spray based on the provided parameters and integrates the satellite motion.
    """
    logM, Rs, q, dirx, diry, dirz, logm, rs, x0, y0, z0, vx0, vy0, vz0, time, alpha = params
    backward_trajectory = integrate_satellite(x0, y0, z0, vx0, vy0, vz0, logM, Rs, q, dirx, diry, dirz, -time, n_steps)

    forward_trajectory  = integrate_satellite(*backward_trajectory[-1, :], logM, Rs, q, dirx, diry, dirz, time*alpha, n_steps)

    theta_sat_forward = jnp.arctan2(forward_trajectory[:, 1], forward_trajectory[:, 0])
    theta_sat_forward = jnp.where(theta_sat_forward < 0, theta_sat_forward + 2 * jnp.pi, theta_sat_forward)
    theta_sat_forward = jax_unwrap(theta_sat_forward)

    hessians  = jax.vmap(NFWHessian, in_axes=(0, 0, 0, None, None, None, None, None, None)) \
                        (forward_trajectory[:, 0], forward_trajectory[:, 1], forward_trajectory[:, 2], logM, Rs, q, dirx, diry, dirz)
    rj, vj, R = get_rj_vj_R(hessians, forward_trajectory, 10 ** logm)
    ic_particle_spray = create_ic_particle_spray(forward_trajectory, rj, vj, R, tail, seed, n_particles, n_steps)

    index = jnp.repeat(jnp.arange(0, n_steps, 1), n_particles // n_steps)
    theta_stream , xv_stream = jax.vmap(integrate_stream_spray, in_axes=(0, 0, 0, 0, 0, 0, 0, None, None, None, None, None, None, None, None, None, None, None)) \
        (index, ic_particle_spray[:, 0], ic_particle_spray[:, 1], ic_particle_spray[:, 2], ic_particle_spray[:, 3], ic_particle_spray[:, 4], ic_particle_spray[:, 5],
        theta_sat_forward, forward_trajectory, logM, Rs, q, dirx, diry, dirz, logm, rs, time)
    
    return theta_stream, xv_stream