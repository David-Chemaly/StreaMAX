import jax
import jax.numpy as jnp
from functools import partial

from .potentials import *
from .utils import get_rj_vj_R, create_ic_particle_spray
from .integrants import integrate_leapfrog_final, integrate_leapfrog_traj, evolve_stream

@partial(jax.jit, static_argnames=('type_host', 'type_sat', 'n_particles', 'n_steps', 'unroll'))
def generate_stream(xv_f, 
                    type_host, params_host, 
                    type_sat, params_sat, 
                    time, alpha, n_steps,
                    n_particles, 
                    unroll,
                    m_f_sat=0, tail=0, seed=111):

    # Define Acceleration and Hessian function from Type of Host
    if type_host == 'PointMass':
        acc_host = PointMass_acceleration
        hessian_host = PointMass_hessian
    elif type_host == 'Isochrone':
        acc_host = Isochrone_acceleration
        hessian_host = Isochrone_hessian
    elif type_host == 'Plummer':
        acc_host = Plummer_acceleration
        hessian_host = Plummer_hessian
    elif type_host == 'NFW':
        acc_host = NFW_acceleration
        hessian_host = NFW_hessian

    # Define Acceleration function from Type of Sat
    if type_sat == 'PointMass':
        acc_sat = PointMass_acceleration
    elif type_sat == 'Isochrone':
        acc_sat = Isochrone_acceleration
    elif type_sat == 'Plummer':
        acc_sat = Plummer_acceleration
    elif type_sat == 'NFW':
        acc_sat = NFW_acceleration

    # Define time step (dt) from total time and steps
    dt = time/n_steps

    # Get initial position of the prog by integrating backwards
    _, xv_i = integrate_leapfrog_final(xv_f, params_host, acc_host, n_steps, dt=-dt, unroll=unroll)

    # Get the orbit of the prog by integrating forwards dt*alpha
    t_sat, xv_sat = integrate_leapfrog_traj(xv_i, params_host, acc_host, n_steps, dt = dt*alpha, unroll=unroll)

    # Compute the Hessian at each point along the orbit
    hessians = jax.vmap(hessian_host, in_axes=(0, 0, 0, None))(xv_sat[:, 0], xv_sat[:, 1], xv_sat[:, 2], params_host)

    # Get the RJ and VJ matrices along the orbit
    m_sat = jnp.linspace(params_sat['logM'], m_f_sat, len(xv_sat))
    rj, vj, R = get_rj_vj_R(hessians, xv_sat, m_sat)

    # Create initial conditions for the particle spray
    ic_particle_spray = create_ic_particle_spray(xv_sat, rj, vj, R, 
                                                    n_particles=n_particles, n_steps=len(xv_sat), tail=tail, seed=seed)

    # Prepare for evolve_stream
    n_groups = n_steps + 1
    n_per_group = n_particles // n_groups
    
    # Reshape particles: (n_groups, n_per_group, 6)
    particles_reshaped = ic_particle_spray.reshape(n_groups, n_per_group, 6)
    
    # Reshape progenitor states: (n_groups, 6) -> (n_groups, n_per_group, 6)
    prog_reshaped = jnp.repeat(xv_sat[:, None, :], n_per_group, axis=1)
    
    # Reshape progenitor mass: (n_groups,) -> (n_groups, n_per_group, 1)
    mass_reshaped = jnp.repeat(m_sat[:, None], n_per_group, axis=1)[..., None]
    
    # Initial g: (n_groups, n_per_group, 1)
    g_init = jnp.zeros((n_groups, n_per_group, 1))
    
    # Construct all_states: (n_groups, n_per_group, 14)
    # r(3), v(3), rp(3), vp(3), g(1), m(1)
    # particles_reshaped has r, v
    # prog_reshaped has rp, vp
    all_states = jnp.concatenate([particles_reshaped, prog_reshaped, g_init, mass_reshaped], axis=-1)
    
    # Steps per group: Group i (released at step i) needs n_steps - i steps
    steps_per_group = jnp.arange(n_steps, -1, -1)
    
    # Constant dm (per step)
    # Total mass change = params_sat['logM'] - m_f_sat
    # Over n_steps steps
    dm = (params_sat['logM'] - m_f_sat) / n_steps

    final_states = evolve_stream(all_states, steps_per_group, dt*alpha, dm,
                                 params_host, params_prog=params_sat,
                                 acc_fn_host=acc_host, acc_fn_prog=acc_sat,
                                 n_steps_total=n_steps, unroll=unroll)
    
    # Flatten results
    final_states_flat = final_states.reshape(n_particles, 14)
    
    xv_stream = final_states_flat[:, :6]
    xhi_stream = final_states_flat[:, 12]

    return t_sat, xv_sat, xv_stream, xhi_stream