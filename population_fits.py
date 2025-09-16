import os
import corner
import pickle
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
from scipy.stats import norm

import jax
import jax.numpy as jnp

import dynesty
import dynesty.utils as dyut

from utils import get_q

BAD_VAL = -1e10

### Uniform ###
def uniform(x, a, delta):
    prob = np.where((x >= a-delta) & (x <= a+delta), 1./delta, 0.)
    return prob

def prior_transform_uniform(p):
    a0, delta0 = p

    a1 = 2*a0
    delta1 = 2*delta0

    return [a1, delta1]

### Gaussian ###
def gaussian(x, mu, sigma):
    return 1/jnp.sqrt(2*jnp.pi*sigma**2) * jnp.exp(-0.5*(x-mu)**2/sigma**2)

def prior_transform_gaussian(p):
    mu0, sigma0 = p

    mu1    = 2*mu0
    sigma1 = 2*sigma0

    return [mu1, sigma1]

### Binomial ###
def binomial(x, mu1, mu2, sigma1, sigma2, prob=0.5):
    return prob * (1/np.sqrt(2*np.pi*sigma1**2) * np.exp(-0.5*(x-mu1)**2/sigma1**2)) + \
           (1-prob) * (1/np.sqrt(2*np.pi*sigma2**2) * np.exp(-0.5*(x-mu2)**2/sigma2**2))

def prior_transform_binomial(p):
    mu1, mu2, sigma1, sigma2, prob = p

    mu1    = mu1
    sigma1 = 2*sigma1
    mu2    = mu2+1
    sigma2 = 2*sigma2
    prob   = prob

    return [mu1, mu2, sigma1, sigma2, prob]

### Likelihood ###
def log_likelihood(theta, dict_data, pop_type='uniform'):
    if pop_type == 'uniform':
        pop_dist = uniform
    elif pop_type == 'gaussian':
        pop_dist = gaussian
    elif pop_type == 'binomial':
        pop_dist = binomial

    # Log-likelihood
    log_likelihood = 0
    for i in range(len(dict_data)):
        likelihood      = np.mean(pop_dist(dict_data[i], *theta))
        if likelihood <= 0:
            return BAD_VAL
        log_likelihood += np.log(likelihood)

    return log_likelihood

### Dynesty Fit ###
def dynesty_fit(dict_data, ndim=2, nlive=500, pop_type='uniform'):
    if pop_type == 'uniform':
        prior_transform = prior_transform_uniform
    elif pop_type == 'gaussian':
        prior_transform = prior_transform_gaussian
    elif pop_type == 'binomial':
        prior_transform = prior_transform_binomial

    nthreads = os.cpu_count()
    mp.set_start_method("spawn", force=True)
    with mp.Pool(nthreads) as poo:
        dns = dynesty.DynamicNestedSampler(log_likelihood,
                                prior_transform,
                                ndim,
                                logl_args=(dict_data, pop_type),
                                nlive=nlive,
                                sample='unif',  
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

### Sampling ###
def subset_as_uniform(u, a, delta, seed=42, N=None):
    rng = np.random.default_rng(seed)
    accept = np.where((u >= a - delta) & (u <= a + delta))[0]
    if N is not None:
        accept = rng.choice(accept, size=N, replace=False)
    return accept

def subset_as_gaussian(u, mu, sigma, seed=42, N=None):
    rng = np.random.default_rng(seed)
    w = norm.pdf(u, loc=mu, scale=sigma)
    p = w / w.max()
    if N is not None:
        accept = []
        while len(accept) != N:
            accept = np.where(rng.random(len(u)) < p)[0]
        return accept

    accept = np.where(rng.random(len(u)) < p)[0]
    return accept

def subset_as_binomial(u, mu1=0.8, mu2=1.2, sigma1=0.1, sigma2=0.1, seed=42, N=None, prob=0.5):
    rng = np.random.default_rng(seed)
    # component densities
    w1 = norm.pdf(u, mu1, sigma1)
    w2 = norm.pdf(u, mu2, sigma2)
    mix = prob*w1 + (1-prob)*w2  # proposal is Uniform(0.5,1.5) with pdf=1
    p = mix / mix.max()

    if N is not None:
        accept = []
        while len(accept) != N:
            accept = np.where(rng.random(len(u)) < p)[0]
        return accept

    accept = np.where(rng.random(len(u)) < p)[0]
    return accept


if __name__ == "__main__":
    true_dist = 'gaussian'
    true_a = 0.8
    true_b = 0.1
    true_c = 0.
    true_d = 0.
    fit_dist = 'binomial'
    fit_type = 'BtoG'
    N_pop = 30
    seed = 1
    ndim = 5
    nlive = 1000

    sigma = 2
    nlive = 2000
    N_streams = 100
    seeds = np.arange(N_streams)
    path = './MockStreams'

    q_true, q_fits = [], []
    for seed in seeds:
        path_seed = os.path.join(path, f'seed{seed}')
        if os.path.exists(os.path.join(path_seed,  f'dict_results_nlive{nlive}_sigma{sigma}.pkl')):
            with open(os.path.join(path_seed, f'dict_results_nlive{nlive}_sigma{sigma}.pkl'), "rb") as f:
                dict_results = pickle.load(f)
            with open(os.path.join(path_seed, f'dict_stream.pkl'), "rb") as f:
                dict_stream = pickle.load(f)
            q_fits.append(get_q(*dict_results['samps'][:, 2:5].T))
            q_true.append(dict_stream['params'][2])
    q_true = np.array(q_true)

    if true_dist == 'uniform':
        arg_take = subset_as_uniform(q_true, true_a, true_b, seed=seed, N=N_pop)
    elif true_dist =='gaussian':
        arg_take = subset_as_gaussian(q_true, true_a, true_b, seed=seed, N=N_pop)
    elif true_dist == 'binomial':
        arg_take = subset_as_binomial(q_true, true_a, true_b, true_c, true_d, seed=seed, N=N_pop)

    q_true = q_true[arg_take]
    new_q_fits = []
    for arg in arg_take:
        new_q_fits.append(q_fits[arg])
    q_fits = new_q_fits

    if fit_dist == 'uniform':
        labels = [r'$a$', r'$b$']
        print(f'Fitting {len(q_true)} streams with [{np.mean(q_true):.2f}, {abs(np.max(q_true)-np.min(q_true))/2:.2f}]')
    elif fit_dist == 'gaussian':
        labels = [r'$\mu$', r'$\sigma$']
        print(f'Fitting {len(q_true)} streams with {np.mean(q_true):.2f} +/- {np.std(q_true):.2f}')
    elif fit_dist == 'binomial':
        labels = [r'$\mu_1$', r'$\mu_2$', r'$\sigma_1$', r'$\sigma_2$']
        print(f'Fitting {len(q_true)} streams with {np.mean(q_true[q_true<1.0]):.2f} +/- {np.std(q_true[q_true<1.0]):.2f} and \
                    {np.mean(q_true[q_true>=1.0]):.2f} +/- {np.std(q_true[q_true>=1.0]):.2f} instead of {true_a:.2f} +/- {true_b:.2f}')

    dict_results = dynesty_fit(q_fits, ndim=ndim, nlive=nlive, pop_type=fit_dist)
    with open(os.path.join(path, f'dict_pop_nlive{nlive}_sigma{sigma}_N{len(q_true)}_'+fit_type+f'_{true_a}-{true_b}.pkl'), 'wb') as f:
        pickle.dump(dict_results, f)

    # Plots the corner plots
    figure = corner.corner(dict_results['samps'], 
            labels=labels,
            color='blue',
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True, 
            title_kwargs={"fontsize": 16},
            truths=[true_a, true_b], 
            truth_color='red')
    figure.savefig(os.path.join(path, f'corner_plot_nlive{nlive}_sigma{sigma}_N{len(q_true)}_'+fit_type+f'_{true_a}-{true_b}.pdf'))
    plt.close(figure)