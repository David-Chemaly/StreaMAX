import os
import pickle
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import dynesty
import dynesty.utils as dyut

from spray import generate_stream_spray
from likelihoods import log_likelihood_spray_base, data_log_likelihood_spray_base
from priors import prior_transform
from utils import get_q, get_track

import corner

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})


def dynesty_fit(dict_data, ndim=14, nlive=2000, sigma=2):
    nthreads = os.cpu_count()
    mp.set_start_method("spawn", force=True)
    with mp.Pool(nthreads) as poo:
        dns = dynesty.DynamicNestedSampler(data_log_likelihood_spray_base,
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
    ndim  = 13
    nlive = 2000

    PATH_DATA = f'/data/dc824-2/SGA_Streams'
    names = np.loadtxt(f'{PATH_DATA}/names.txt', dtype=str)

    for name in tqdm(names[2:], leave=True):
        if not os.path.exists(os.path.join(PATH_DATA,  f'running_nlive{nlive}_fixedProgcenter.txt')):
            np.savetxt(os.path.join(PATH_DATA,  f'running_nlive{nlive}_fixedProgcenter.txt'), [1])

            with open(f"{PATH_DATA}/{name}/dict_track.pkl", "rb") as f:
                dict_data = pickle.load(f)
            
            # This sets the progenitor in the middle of the stream
            dict_data['theta'] -= np.median(dict_data['theta'])

            print(f'Fitting {name} with nlive={nlive} and fixed progenitor at center')
            dict_results = dynesty_fit(dict_data, ndim=ndim, nlive=nlive)
            with open(f'{PATH_DATA}/{name}/dict_results_nlive{nlive}_fixedProgcenter.pkl', 'wb') as f:
                pickle.dump(dict_results, f)

            # Plot and Save corner plot
            labels = ['logM', 'Rs', 'dirx', 'diry', 'dirz', 'logm', 'rs', 'x0', 'z0', 'vx0', 'vy0', 'vz0', 'time']
            figure = corner.corner(dict_results['samps'], 
                        labels=labels,
                        color='blue',
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True, 
                        title_kwargs={"fontsize": 16})
            figure.savefig(f'{PATH_DATA}/{name}/corner_plot_nlive{nlive}_fixedProgcenter.pdf')
            plt.close(figure)

            # Plot and Save flattening
            q_samps = get_q(dict_results['samps'][:, 2], dict_results['samps'][:, 3], dict_results['samps'][:, 4])
            plt.figure(figsize=(8, 6))
            plt.hist(q_samps, bins=30, density=True, alpha=0.7, color='blue')
            plt.xlabel('Halo Flattening')
            plt.ylabel('Density')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{PATH_DATA}/{name}/q_posterior_nlive{nlive}_fixedProgcenter.pdf')
            plt.close()

            # Plot and Save best fit
            plt.figure(figsize=(18, 7))
            plt.subplot(1, 2, 1)
            plt.xlabel(r'x [kpc]')
            plt.ylabel(r'y [kpc]')

            best_params = dict_results['samps'][np.argmax(dict_results['logl'])]
            q_best = get_q(best_params[2], best_params[3], best_params[4])
            best_params = np.concatenate([best_params[:2], [q_best], best_params[2:8], [0.], best_params[8:], [1.]])
            np.savetxt(f'{PATH_DATA}/{name}/best_params_nlive{nlive}_fixedProgcenter.txt', best_params)

            theta_stream, xv_stream, theta_sat, xv_sat = generate_stream_spray(best_params, seed=111)
            _, theta_bin, r_bin, _ = get_track(theta_stream, xv_stream[:, 0], xv_stream[:, 1])
            x_bin = r_bin * np.cos(theta_bin)
            y_bin = r_bin * np.sin(theta_bin)

            plt.scatter(xv_stream[:, 0], xv_stream[:, 1], s=0.1, cmap='seismic', c=theta_stream, vmin=-2*np.pi, vmax=2*np.pi)
            plt.scatter(x_bin, y_bin, c='lime')
            plt.scatter(dict_data['x'], dict_data['y'], c='red')
            plt.axvline(0, color='k', linestyle='--', lw=1, c='gray')
            plt.axhline(0, color='k', linestyle='--', lw=1, c='gray')
            plt.axis('equal')

            plt.subplot(1, 2, 2)
            r_stream = np.sqrt(xv_stream[:, 0]**2 + xv_stream[:, 1]**2)
            plt.scatter(theta_stream, r_stream, s=0.1, cmap='seismic', c=theta_stream, vmin=-2*np.pi, vmax=2*np.pi, label='Stream Model')
            plt.scatter(theta_bin, r_bin, c='lime', label='Medians')
            plt.colorbar(label='Angle (rad)')
            plt.scatter(dict_data['theta'], dict_data['r'], c='red', label='Data')
            plt.xlabel('Angle (rad)')
            plt.ylabel('Radius (kpc)')
            plt.legend(loc='best')

            plt.tight_layout()
            plt.savefig(f'{PATH_DATA}/{name}/best_fit_nlive{nlive}_fixedProgcenter.pdf')
            plt.close()
