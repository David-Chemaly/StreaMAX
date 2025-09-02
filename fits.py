import os
import pickle
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import dynesty
import dynesty.utils as dyut

from spray import generate_stream_spray
from likelihoods import log_likelihood
from priors import prior_transform
from utils import get_q, get_track

import corner

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

def dynesty_fit(dict_data, ndim=14, nlive=2000, sigma=2):
    nthreads = os.cpu_count()
    mp.set_start_method("spawn", force=True)
    with mp.Pool(nthreads) as poo:
        dns = dynesty.DynamicNestedSampler(log_likelihood,
                                prior_transform,
                                ndim,
                                logl_args=(dict_data, sigma),
                                nlive=nlive,
                                sample='rslice',
                                pool=poo,
                                queue_size=nthreads * 2)
        dns.run_nested(n_effective=10000)

    res   = dns.results
    inds  = np.arange(len(res.samples))
    inds  = dyut.resample_equal(inds, weights=np.exp(res.logwt - res.logz[-1]))
    samps = res.samples[inds]
    logl  = res.logl[inds]

    dns_results = {
                    'dns': dns,
                    'samps': samps,
                    'logl': logl,
                    'logz': res.logz,
                    'logzerr': res.logzerr,
                }

    return dns_results


if __name__ == "__main__":
    N = 100
    seeds = np.arange(100)

    ndim  = 13
    nlive = 1000
    sigma = 1

    for seed in tqdm(seeds, leave=True):
        path = f'/data/dc824-2/MockStreams/seed{seed}' 

        if not os.path.exists(os.path.join(path,  f'running_nlive{nlive}_sigma{sigma}.txt')):
            np.savetxt(os.path.join(path,  f'running_nlive{nlive}_sigma{sigma}.txt'), [1])

            # Load data and add noise baised on sigma
            with open(os.path.join(path, "dict_stream.pkl"), "rb") as f:
                    dict_data = pickle.load(f)
            params_data = dict_data['params']
            params_data = np.concatenate([params_data[:2], params_data[3:9], params_data[10:-1]])

            r_sig = dict_data['r_bin'] * sigma / 100
            rng   = np.random.default_rng(int(seed))
            r_err = rng.normal(0, r_sig)

            dict_data['r_bin'] += r_err
            dict_data['r_sig'] = r_sig
            dict_data['x_bin'] = dict_data['r_bin'] * np.cos(dict_data['theta_bin'])
            dict_data['y_bin'] = dict_data['r_bin'] * np.sin(dict_data['theta_bin'])
            print(log_likelihood(params_data, dict_data))

            # Fit with dynesty
            dict_results = dynesty_fit(dict_data, ndim=ndim, nlive=nlive)
            with open(os.path.join(path, f'dict_results_nlive{nlive}_sigma{sigma}.pkl'), 'wb') as f:
                pickle.dump(dict_results, f)

            # Plot and Save corner plot
            labels = ['logM', 'Rs', 'dirx', 'diry', 'dirz', 'logm', 'rs', 'x0', 'z0', 'vx0', 'vy0', 'vz0', 'time']
            figure = corner.corner(dict_results['samps'], 
                        labels=labels,
                        color='blue',
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True, 
                        title_kwargs={"fontsize": 16},
                        truths=params_data, 
                        truth_color='red')
            figure.savefig(os.path.join(path, f'corner_plot_nlive{nlive}_sigma{sigma}.pdf'))
            plt.close(figure)

            # Plot and Save flattening
            q_samps = get_q(dict_results['samps'][:, 2], dict_results['samps'][:, 3], dict_results['samps'][:, 4])
            plt.figure(figsize=(8, 6))
            plt.hist(q_samps, bins=30, density=True, alpha=0.7, color='blue')
            plt.axvline(dict_data['params'][2], color='red', linestyle='--', label='True Value')
            plt.xlabel('Halo Flattening')
            plt.ylabel('Density')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(path, f'q_posterior_nlive{nlive}_sigma{sigma}.pdf'))
            plt.close()

            # Plot and Save best fit
            plt.figure(figsize=(18, 7))
            plt.subplot(1, 2, 1)
            plt.xlabel(r'x [kpc]')
            plt.ylabel(r'y [kpc]')

            best_params = dict_results['samps'][np.argmax(dict_results['logl'])]
            q_best = get_q(best_params[2], best_params[3], best_params[4])
            best_params = np.concatenate([best_params[:2], [q_best], best_params[2:8], [0.], best_params[8:], [1.]])
            np.savetxt(os.path.join(path, f'best_params_nlive{nlive}_sigma{sigma}.txt'), best_params)

            theta_stream, xv_stream, theta_sat, xv_sat = generate_stream_spray(best_params, seed=111)
            _, theta_bin, r_bin, _ = get_track(theta_stream, xv_stream[:, 0], xv_stream[:, 1])
            x_bin = r_bin * np.cos(theta_bin)
            y_bin = r_bin * np.sin(theta_bin)

            plt.scatter(xv_stream[:, 0], xv_stream[:, 1], s=0.1, cmap='seismic', c=theta_stream, vmin=-2*np.pi, vmax=2*np.pi)
            plt.scatter(x_bin, y_bin, c='lime')
            plt.scatter(dict_data['x_bin'], dict_data['y_bin'], c='red')
            plt.axvline(0, color='k', linestyle='--', lw=1, c='gray')
            plt.axhline(0, color='k', linestyle='--', lw=1, c='gray')
            plt.axis('equal')

            plt.subplot(1, 2, 2)
            r_stream = np.sqrt(xv_stream[:, 0]**2 + xv_stream[:, 1]**2)
            plt.scatter(theta_stream, r_stream, s=0.1, cmap='seismic', c=theta_stream, vmin=-2*np.pi, vmax=2*np.pi, label='Stream Model')
            plt.scatter(theta_bin, r_bin, c='lime', label='Medians')
            plt.colorbar(label='Angle (rad)')
            plt.scatter(dict_data['theta_bin'], dict_data['r_bin'], c='red', label='Data')
            plt.xlabel('Angle (rad)')
            plt.ylabel('Radius (kpc)')
            plt.legend(loc='best')

            plt.tight_layout()
            plt.savefig(os.path.join(path, f'best_fit_nlive{nlive}_sigma{sigma}.pdf'))
            plt.close()
