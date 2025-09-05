import os
import pickle
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
from scipy.stats import norm

import dynesty
import dynesty.utils as dyut

from utils import get_q

#################################
# Uniform Population Fits
#################################
def uniform(x, a, b):
    return np.where((x >= a) & (x <= b), 1/(b-a), 0)

def prior_transform_uniform(p):
    a0, b0 = p

    a1 = 2*a0
    b1 = 2*b0

    return [a1, b1]

def log_likelihood_uniform(theta, q_fits):
    a, b = theta

    # Log-likelihood
    log_likelihood = 0
    for i in range(len(q_fits)):
        likelihood      = np.mean(uniform(q_fits[i], a, b))
        log_likelihood += np.log(likelihood)

    return log_likelihood

def dynesty_fit_uniform(dict_data, ndim=2, nlive=500):
    nthreads = os.cpu_count()
    with mp.Pool(nthreads) as poo:
        dns = dynesty.DynamicNestedSampler(log_likelihood_uniform,
                                prior_transform_uniform,
                                ndim,
                                logl_args=(dict_data, ),
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

#################################
# Gaussian Population Fits
#################################
def gaussian(x, mu, sigma):
    return 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-0.5*(x-mu)**2/sigma**2)

def prior_transform_gaussian(p):
    mu0, sigma0 = p

    mu1    = 2*mu0
    sigma1 = 2*sigma0

    return [mu1, sigma1]

def log_likelihood_gaussian(theta, q_fits):
    mu, sigma = theta

    # Log-likelihood
    log_likelihood = 0
    for i in range(len(q_fits)):
        likelihood      = np.mean(gaussian(q_fits[i], mu, sigma))
        log_likelihood += np.log(likelihood)

    return log_likelihood

def dynesty_fit_gaussian(dict_data, ndim=2, nlive=500):
    nthreads = os.cpu_count()
    with mp.Pool(nthreads) as poo:
        dns = dynesty.DynamicNestedSampler(log_likelihood_gaussian,
                                prior_transform_gaussian,
                                ndim,
                                logl_args=(dict_data, ),
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

#################################
# Binomial Population Fits
#################################
def binomial(x, mu1, mu2, sigma1, sigma2, prob=0.5):
    return prob * (1/np.sqrt(2*np.pi*sigma1**2) * np.exp(-0.5*(x-mu1)**2/sigma1**2)) + \
           (1-prob) * (1/np.sqrt(2*np.pi*sigma2**2) * np.exp(-0.5*(x-mu2)**2/sigma2**2))

def prior_transform_binomial(p):
    mu1, mu2, sigma1, sigma2 = p

    mu1    = 2*mu1
    sigma1 = 2*sigma1
    mu2    = 2*mu2
    sigma2 = 2*sigma2
    # prob   = 1 - prob

    return [mu1, mu2, sigma1, sigma2]

def log_likelihood_binomial(theta, q_fits):
    mu1, mu2, sigma1, sigma2 = theta

    # Log-likelihood
    log_likelihood = 0
    for i in range(len(q_fits)):
        likelihood      = np.mean(binomial(q_fits[i], mu1, mu2, sigma1, sigma2))
        log_likelihood += np.log(likelihood)

    return log_likelihood

def dynesty_fit_binomial(dict_data, ndim=4, nlive=500):
    nthreads = os.cpu_count()
    with mp.Pool(nthreads) as poo:
        dns = dynesty.DynamicNestedSampler(log_likelihood_binomial,
                                prior_transform_binomial,
                                ndim,
                                logl_args=(dict_data, ),
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

def subset_as_gaussian(u, mu, sigma, N=None):
    w = norm.pdf(u, loc=mu, scale=sigma)
    p = w / w.max()
    if N is not None:
        accept = []
        while len(accept) != N:
            accept = np.where(np.random.rand(len(u)) < p)[0]
        return accept

    accept = np.where(np.random.rand(len(u)) < p)[0]
    return accept

def subset_as_binomial(u, mu1=0.8, mu2=1.2, sigma1=0.1, sigma2=0.1, N=None, prob=0.5):
    # component densities
    w1 = norm.pdf(u, mu1, sigma1)
    w2 = norm.pdf(u, mu2, sigma2)
    mix = prob*w1 + (1-prob)*w2  # proposal is Uniform(0.5,1.5) with pdf=1
    p = mix / mix.max()

    if N is not None:
        accept = []
        while len(accept) != N:
            accept = np.where(np.random.rand(len(u)) < p)[0]
        return accept

    accept = np.where(np.random.rand(len(u)) < p)[0]
    return accept

if __name__ == "__main__":
    true_dist = 'uniform'
    true_a = 0.5
    true_b = 1.5
    true_c = 0.
    true_d = 0.
    fit_dist = 'binomial'
    fit_type = 'BtoU'
    N_pop = None

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
        arg_take = np.where((q_true >= true_a) & (q_true <= true_b))[0]
    elif true_dist =='gaussian':
        arg_take = subset_as_gaussian(q_true, true_a, true_b)
    elif true_dist == 'binomial':
        arg_take = subset_as_binomial(q_true, true_a, true_b, true_c, true_d)

    q_true = q_true[arg_take]
    new_q_fits = []
    for arg in arg_take:
        new_q_fits.append(q_fits[arg])
    q_fits = new_q_fits

    if fit_dist == 'uniform':
        dict_results = dynesty_fit_uniform(q_fits, ndim=2, nlive=500)
    elif fit_dist == 'gaussian':
        dict_results = dynesty_fit_gaussian(q_fits, ndim=2, nlive=500)
    elif fit_dist == 'binomial':
        dict_results = dynesty_fit_binomial(q_fits, ndim=4, nlive=1000)
    with open(os.path.join('./MockStreams', f'dict_pop_nlive{nlive}_sigma{sigma}_N{len(q_true)}_'+fit_type+'.pkl'), 'wb') as f:
        pickle.dump(dict_results, f)

    # Plots
    