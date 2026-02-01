import jax
import jax.numpy as jnp
from functools import partial

from .potentials import *
from .utils import get_rj_vj_R
from .methods import create_ic_particle_spray_Fardal2015
from .integrants import integrate_leapfrog_final, integrate_leapfrog_traj, combined_integrate_leapfrog_final

@partial(jax.jit, static_argnames=('type_host', 'type_sat', 'n_particles', 'n_steps', 'unroll', 
                                    'type_method'))
def generate_stream(xv_f, 
                    type_host, params_host, 
                    type_sat, params_sat, 
                    time, alpha, n_steps,
                    n_particles, 
                    unroll,
                    type_method='Fardal2015',
                    m_f_sat=0, 
                    tail=0, seed=111):

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
    elif type_host == 'MiyamotoNagai':
        acc_host = MiyamotoNagai_acceleration
        hessian_host = MiyamotoNagai_hessian
    elif type_host == 'Logarithmic':
        acc_host = Logarithmic_acceleration
        hessian_host = Logarithmic_hessian
    elif type_host == 'Bar_potential':
        acc_host = Bar_acceleration
        hessian_host = Bar_hessian
    elif type_host == 'Hernquist':
        acc_host = Hernquist_acceleration
        hessian_host = Hernquist_hessian
    elif type_host == 'Logarithmic_potential':
        acc_host = Logarithmic_acceleration
        hessian_host = Logarithmic_hessian
    elif type_host == 'ExpDisk_potential':
        acc_host = ExpDisk_acceleration
        hessian_host = ExpDisk_hessian
    elif type_host == 'Composite':
        # TODO: implement composite potential selection
        raise NotImplementedError("Composite potential not yet implemented.")
    else:
        raise NotImplementedError(f"Host potential {type_host} not implemented.")

    # Define Acceleration function from Type of Sat
    if type_sat == 'PointMass':
        acc_sat = PointMass_acceleration
    elif type_sat == 'Isochrone':
        acc_sat = Isochrone_acceleration
    elif type_sat == 'Plummer':
        acc_sat = Plummer_acceleration
    elif type_sat == 'NFW':
        acc_sat = NFW_acceleration
    else:
        raise NotImplementedError(f"Satellite potential {type_sat} not implemented.")

    # Define time step (dt) from total time and steps
    dt = time/n_steps

    # Get initial position of the prog by integrating backwards
    _, xv_i = integrate_leapfrog_final(xv_f, params_host, acc_host, n_steps, dt=-dt, unroll=unroll)

    # Get the orbit of the prog by integrating forwards dt*alpha
    t_sat, xv_sat = integrate_leapfrog_traj(xv_i, params_host, acc_host, n_steps, dt = dt*alpha, unroll=unroll)

    # Compute the Hessian at each point along the orbit
    hessians = jax.vmap(hessian_host, in_axes=(0, 0, 0, None))(xv_sat[:, 0], xv_sat[:, 1], xv_sat[:, 2], params_host)

    # Satellite mass evolution along the orbit
    m_sat = jnp.linspace(params_sat['logM'], m_f_sat, len(xv_sat))
    dm    = (m_sat - m_f_sat)/n_steps # TODO: allow for mass loss with variable dm

    # Create initial conditions for the particle spray
    if type_method == 'Fardal2015':
        # Get the RJ and VJ matrices along the orbit
        rj, vj, R = get_rj_vj_R(hessians, xv_sat, m_sat)

        # Get the particle spray initial conditions
        ic_particle_spray = create_ic_particle_spray_Fardal2015(xv_sat, rj, vj, R, 
                                                    n_particles=n_particles, n_steps=len(xv_sat), tail=tail, seed=seed)
    else:
        raise NotImplementedError(f"Method {type_method} not implemented.")

    # Integrate the particle spray
    index = jnp.repeat(jnp.arange(0, n_steps+1, 1), n_particles // (n_steps+1))
    xv_stream, xhi_stream = jax.vmap(combined_integrate_leapfrog_final, 
                                    in_axes=(0, 0, None, None, None, None, None, None, None, None, None, None)) \
                                    (index, ic_particle_spray, params_host, 
                                    xv_sat, params_sat,
                                    acc_host, acc_sat,
                                    n_steps,
                                    (time-t_sat)*alpha/n_steps,
                                    m_sat,
                                    dm,
                                    unroll)

    return t_sat, xv_sat, xv_stream, xhi_stream