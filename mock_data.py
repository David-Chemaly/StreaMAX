from models import generate_stream_spray
import scipy

import jax
import jax.numpy as jnp
import jax.random as random
from utils import get_q
from priors import sample_params
from utils import get_track

import numpy as np

# @jax.jit
def get_data(seed, sigma=2):
    is_data = False
    rng = np.random.default_rng(seed)

    while not is_data:
        seed_here = rng.integers(0, 2**32-1)
        params = sample_params(seed_here)

        q = get_q(params[2], params[3], params[4])
        theta_stream, xv_stream = generate_stream_spray(params[:2] + (q, ) + params[2:],  seed=111, n_steps=500, n_particles=10000)
        count, theta_bin, r_bin, w_bin = get_track(theta_stream, xv_stream[:, 0], xv_stream[:, 1])

        crit1 = jnp.nanmin(r_bin) > 10  # Must be further than 10kpc minimum
        crit2 = jnp.sum(jnp.where(count > 100, 1, 0)) > 9 # Must have at least 10 bins with more than 100 particles
        crit3 = jnp.nansum(r_bin[:-1]*jnp.tanh(jnp.diff(theta_bin))) > 100 # Must have length of at least 100kpc

        if crit1 and crit2 and crit3:
            is_data = True

    dict_data = {
        'params': params,
        'theta_stream': theta_stream,
        'x_stream': xv_stream[:, 0],
        'y_stream': xv_stream[:, 1],
        'count': count,
        'theta_bin': theta_bin,
        'r_bin': r_bin,
        'w_bin': w_bin
    }

    return dict_data

# def get_data_stream(q_true, seed, sigma=1, tail=0, min_count=11, n_theta_min=13, r_min=20, r_max=500, l_min=200, R2_max=0.8):
#     is_data = False
#     rng = np.random.default_rng(seed)
#     theta_gap = np.diff(np.linspace(-2*np.pi, 2*np.pi, N_BINS))[0]
#     while not is_data:
#         logM, Rs, q, dirx, diry, dirz, logm, rs, x0, y0, z0, vx0, vy0, vz0, time, alpha = sample_params_data(q_true, rng.integers(0, 2**32-1))

#         theta_stream, x_stream, y_stream, vz_stream, r_meds, w_meds, x_meds, y_meds, vz_meds = jax_stream_model(logM, Rs, q, dirx, diry, dirz, logm, rs, x0, y0, z0, vx0, vy0, vz0, time, alpha, tail, min_count)

#         if np.sum(~np.isnan(r_meds)) > n_theta_min:

#             crit1 = np.all(np.diff(np.where(~np.isnan(r_meds))) == 1) # Must be continuous
#             crit2 = np.nanmin(r_meds) > r_min # Must be further than 10kpc minimum
#             crit3 = np.sum(r_meds[~np.isnan(r_meds)][:-1] * np.tan(theta_gap)) > l_min # Must have length of at least 100kpc
#             crit4 = np.corrcoef(x_meds[~np.isnan(x_meds)], y_meds[~np.isnan(y_meds)])[0, 1]**2 <= R2_max # Can't be too much a straight line
#             crit5 = np.nanmax(r_meds) < r_max # Must be less than 500kpc
#             crit6 = np.nanmin((x_stream**2 + y_stream**2)**0.5) > r_min/4

#             if crit1 and crit2 and crit3 and crit4 and crit5 and crit6:
#                 is_data = True

#     params = np.array([logM, Rs, q, dirx, diry, dirz, logm, rs, x0, z0, vx0, vy0, vz0, time, alpha])

#     r_err = np.abs(r_meds)*sigma/100
#     r_noise = rng.normal(0, r_err)

#     w_err = np.abs(w_meds)*sigma/100
#     w_noise = rng.normal(0, w_err)

#     dict_data = {'params': params, 'theta_stream': theta_stream, 'x_stream': x_stream, 'y_stream': y_stream, 'x_meds': x_meds, 'y_meds': y_meds, 'r_data': r_meds+r_noise, 'r_err': r_err, 'r_noise': r_noise, 'w_data': w_meds+w_noise, 'w_err': w_err, 'w_noise': w_noise, 'params_time':time}

#     return dict_data
