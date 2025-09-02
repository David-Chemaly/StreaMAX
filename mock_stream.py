import jax.numpy as jnp
from utils import get_q, get_track
from priors import prior_transform
from spray import generate_stream_spray
from tqdm import tqdm
import numpy as np
import os
import pickle

def get_stream(seed, sigma=2, ndim=13, min_count=10):
    is_data = False
    rng = np.random.default_rng(int(seed))

    p = rng.uniform(0, 1, size=ndim)

    while not is_data:
        xv = rng.uniform(0, 1, size=ndim-3)  # Resample 6D phase space
        p[:2] = xv[:2]
        p[5:] = xv[2:]
        params = prior_transform(p)
        q = get_q(params[2], params[3], params[4])
        params = np.concatenate([params[:2], [q], params[2:8], [0.], params[8:], [1.]])

        theta_stream, xv_stream, theta_sat, xv_sat = generate_stream_spray(params,  seed=111)
        count, theta_bin, r_bin, w_bin = get_track(theta_stream, xv_stream[:, 0], xv_stream[:, 1])
        r_stream = jnp.sqrt(xv_stream[:, 0]**2 + xv_stream[:, 1]**2)

        crit1 = jnp.all(jnp.diff(jnp.where(count > min_count)[0]) == 1) # Must be continuous and
        crit2 = jnp.sum(jnp.where(count > min_count, 1, 0)) > 9   # Must have at least 10 bins with more than 100 particles
        crit3 = jnp.nansum(r_bin[:-1]*jnp.tanh(jnp.diff(theta_bin))) > 100 # Must have length of at least 100kpc
        crit4 = jnp.min(r_stream) > 2  # Must be further than 2kpc minimum
        crit5 = jnp.max(r_stream) < 500  # Must be less than 500kpc
        crit6 = jnp.all(jnp.diff(theta_sat) > 0)  # Must be monotonic

        if crit1 and crit2 and crit3 and crit4 and crit5 and crit6: 
            is_data = True

    dict_stream = {
        'params': params,
        'theta_stream': theta_stream,
        'x_stream': xv_stream[:, 0],
        'y_stream': xv_stream[:, 1],
        'theta_sat': theta_sat,
        'x_sat': xv_sat[:, 0],
        'y_sat': xv_sat[:, 1],
        'count': count,
        'theta_bin': theta_bin,
        'r_bin': r_bin,
        'w_bin': w_bin
    }

    return dict_stream

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 18})

    N = 100
    seeds = jnp.arange(100)

    for seed in tqdm(seeds, leave=True):
        path = f'/data/dc824-2/MockStreams/seed{seed}'
        os.makedirs(path, exist_ok=True)

        dict_stream = get_stream(seed, sigma=2)

        with open(os.path.join(path, 'dict_stream.pkl'), 'wb') as f:
            pickle.dump(dict_stream, f)

        xv_stream = dict_stream['x_stream'], dict_stream['y_stream']

        # Plot the stream
        plt.figure(figsize=(18, 7))
        plt.subplot(1, 2, 1)
        plt.scatter(dict_stream['x_stream'], dict_stream['y_stream'], s=0.1, cmap='seismic', c=dict_stream['theta_stream'], vmin=-2*np.pi, vmax=2*np.pi)
        plt.scatter(dict_stream['x_sat'][-1], dict_stream['y_sat'][-1], s=20, c='black')
        x_bin = dict_stream['r_bin'] * np.cos(dict_stream['theta_bin'])
        y_bin = dict_stream['r_bin'] * np.sin(dict_stream['theta_bin'])
        plt.axvline(0, color='k', linestyle='--', lw=1, c='gray')
        plt.axhline(0, color='k', linestyle='--', lw=1, c='gray')
        plt.xlabel('X (kpc)')
        plt.ylabel('Y (kpc)')
        plt.axis('equal')
        plt.subplot(1, 2, 2)
        r_stream = jnp.sqrt(dict_stream['x_stream']**2 + dict_stream['y_stream']**2)
        r_sat = jnp.sqrt(dict_stream['x_sat']**2 + dict_stream['y_sat']**2)
        plt.scatter(dict_stream['theta_stream'], r_stream, s=0.1, cmap='seismic', c=dict_stream['theta_stream'], vmin=-2*np.pi, vmax=2*np.pi)
        plt.colorbar(label='Angle (rad)')
        plt.plot(dict_stream['theta_bin'], dict_stream['r_bin'], 'o', c='lime', markersize=4)
        plt.xlabel('Angle (rad)')
        plt.ylabel('Radius (kpc)')
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'stream.pdf'))