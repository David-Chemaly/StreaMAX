import os
import pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

from utils import get_q, get_track, inference_first, get_track_from_data
from spray_base import generate_stream_spray_base
from spray import generate_stream_spray
from first import generate_stream_first
from streak import generate_stream_streak

BAD_VAL = -1e10

def data_log_likelihood_spray_base(params, dict_data, seed=111, N_min=11, q_min=0.5, q_max=2.0):
    q      = get_q(params[2], params[3], params[4], q_min=q_min, q_max=q_max)
    params = np.concatenate([params[:2], [q], params[2:8], [0.], params[8:], [1.]])

    theta_stream, xv_stream, _, _ = generate_stream_spray_base(params,  seed)
    count_bin, r_bin, w_bin = get_track_from_data(theta_stream, xv_stream[:, 0], xv_stream[:, 1], dict_data['theta'])

    n_bad = np.sum(count_bin < N_min)
    if np.all(np.isnan(r_bin)):
        logl = BAD_VAL * len(r_bin)
    elif n_bad == 0:
        model_err = w_bin / np.sqrt(count_bin)
        var = model_err**2 + dict_data['r_err']**2
        logl  = -.5 * np.sum(  (r_bin - dict_data['r'])**2 / var  + np.log(2 * np.pi * var)  )
    else:
        logl = BAD_VAL * n_bad

    return logl

def data_log_likelihood_spray_base_regular(params, dict_data, seed=111, N_min=11):
    params = np.concatenate([params[:2], [1., 0., 0., 1.], params[2:5], [0.], params[5:], [1.]])

    theta_stream, xv_stream, _, _ = generate_stream_spray_base(params,  seed)
    count_bin, r_bin, w_bin = get_track_from_data(theta_stream, xv_stream[:, 0], xv_stream[:, 1], dict_data['theta'])
    
    n_bad = np.sum(count_bin < N_min)
    if np.all(np.isnan(r_bin)):
        logl = BAD_VAL * len(r_bin)
    elif n_bad == 0:
        model_err = w_bin / np.sqrt(count_bin)
        var = model_err**2 + dict_data['r_err']**2
        logl  = -.5 * np.sum(  (r_bin - dict_data['r'])**2 / var  + np.log(2 * np.pi * var)  )
    else:
        logl = BAD_VAL * n_bad

    return logl

def log_likelihood_spray_base(params, dict_data, seed=13, min_count=100):
    q      = get_q(params[2], params[3], params[4])
    params = np.concatenate([params[:2], [q], params[2:8], [0.], params[8:], [1.]])

    theta_stream, xv_stream, _, _ = generate_stream_spray_base(params,  seed)
    _, _, r_bin, _ = get_track(theta_stream, xv_stream[:, 0], xv_stream[:, 1])
    

    arg_take = ~np.isnan(dict_data['r_bin']) * (dict_data['count'] > min_count)
    n_bad    = np.sum(np.isnan(r_bin[arg_take]))

    if np.all(np.isnan(r_bin)):
        logl = BAD_VAL * len(r_bin)

    elif n_bad == 0:
        logl  = -.5 * np.sum( ( (r_bin[arg_take] - dict_data['r_bin'][arg_take]) / dict_data['r_sig'][arg_take] )**2 )

    else:
        logl = BAD_VAL * n_bad

    return logl

def log_likelihood_spray(params, dict_data, seed=13, min_count=100):
    q      = get_q(params[2], params[3], params[4])
    params = np.concatenate([params[:2], [q], params[2:8], [0.], params[8:], [1.]])

    theta_stream, xv_stream, _, _ = generate_stream_spray(params,  seed)
    _, _, r_bin, _ = get_track(theta_stream, xv_stream[:, 0], xv_stream[:, 1])
    

    arg_take = ~np.isnan(dict_data['r_bin']) * (dict_data['count'] > min_count)
    n_bad    = np.sum(np.isnan(r_bin[arg_take]))

    if np.all(np.isnan(r_bin)):
        logl = BAD_VAL * len(r_bin)

    elif n_bad == 0:
        logl  = -.5 * np.sum( ( (r_bin[arg_take] - dict_data['r_bin'][arg_take]) / dict_data['r_sig'][arg_take] )**2 )

    else:
        logl = BAD_VAL * n_bad

    return logl

def log_likelihood_first(params, dict_data, seed=13, min_count=100):
    q      = get_q(params[2], params[3], params[4])
    params = np.concatenate([params[:2], [q], params[2:8], [0.], params[8:], [1.]])

    theta_stream_first, xv_stream_first, _, _, S, _, refs, _ = generate_stream_first(params,  seed=seed)
    theta_stream, xv_stream = inference_first(theta_stream_first, xv_stream_first, refs, S, seed=seed, disp_x=0.1, disp_v=1.)

    _, _, r_bin, _ = get_track(theta_stream, xv_stream[:, 0], xv_stream[:, 1])
    

    arg_take = ~np.isnan(dict_data['r_bin']) * (dict_data['count'] > min_count)
    n_bad    = np.sum(np.isnan(r_bin[arg_take]))

    if np.all(np.isnan(r_bin)):
        logl = BAD_VAL * len(r_bin)

    elif n_bad == 0:
        logl  = -.5 * np.sum( ( (r_bin[arg_take] - dict_data['r_bin'][arg_take]) / dict_data['r_sig'][arg_take] )**2 )

    else:
        logl = BAD_VAL * n_bad

    return logl

def log_likelihood_streak(params, dict_data, seed=13, min_count=100):
    q      = get_q(params[2], params[3], params[4])
    params = np.concatenate([params[:2], [q], params[2:8], [0.], params[8:], [1.]])

    theta_stream, xv_stream, _, _ = generate_stream_streak(params,  seed)

    _, _, r_bin, _ = get_track(theta_stream, xv_stream[:, 0], xv_stream[:, 1])
    

    arg_take = ~np.isnan(dict_data['r_bin']) * (dict_data['count'] > min_count)
    n_bad    = np.sum(np.isnan(r_bin[arg_take]))

    if np.all(np.isnan(r_bin)):
        logl = BAD_VAL * len(r_bin)

    elif n_bad == 0:
        logl  = -.5 * np.sum( ( (r_bin[arg_take] - dict_data['r_bin'][arg_take]) / dict_data['r_sig'][arg_take] )**2 )

    else:
        logl = BAD_VAL * n_bad

    return logl

if __name__ == "__main__":
    N = 100
    seeds = np.arange(100)

    ndim  = 13
    nlive = 1000
    sigma = 2

    logl_spray_base, logl_spray, logl_streak, logl_first = [], [], [], []
    dev_spray_base, dev_spray, dev_streak, dev_first = [], [], [], []
    for seed in tqdm(seeds):
        path = f'./MockStreams/seed{seed}' 

        # if not os.path.exists(os.path.join(path,  f'running_nlive{nlive}_sigma{sigma}.txt')):
        #     np.savetxt(os.path.join(path,  f'running_nlive{nlive}_sigma{sigma}.txt'), [1])

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

        N = 1
        # spray_base, spray, streak, first = [], [], [], []
        spray_base, spray, streak, first = 0., 0., 0., 0.
        for n in range(N):
            spray_base += log_likelihood_spray_base(params_data, dict_data, seed=np.random.randint(1, 1000000), min_count=100)
            spray += log_likelihood_spray(params_data, dict_data, seed=np.random.randint(1, 1000000), min_count=100)
            streak += log_likelihood_streak(params_data, dict_data, seed=np.random.randint(1, 1000000), min_count=100)
            first += log_likelihood_first(params_data, dict_data, seed=np.random.randint(1, 1000000), min_count=100)

            # spray_base.append(log_likelihood_spray_base(params_data, dict_data, seed=np.random.randint(1, 1000000), min_count=100))
            # spray.append(log_likelihood_spray(params_data, dict_data, seed=np.random.randint(1, 1000000), min_count=100))
            # streak.append(log_likelihood_streak(params_data, dict_data, seed=np.random.randint(1, 1000000), min_count=100))
            # first.append(log_likelihood_first(params_data, dict_data, seed=np.random.randint(1, 1000000), min_count=100))
        # spray_base = np.array(spray_base)
        # spray = np.array(spray)
        # streak = np.array(streak)
        # first = np.array(first)
        spray_base /= N
        spray /= N
        streak /= N
        first /= N

        dev_spray.append(abs(spray - spray_base))
        dev_streak.append(abs(streak - spray_base))
        dev_first.append(abs(first - spray_base))
    dev_spray = np.array(dev_spray)
    dev_streak = np.array(dev_streak)
    dev_first = np.array(dev_first)

    plt.figure(figsize=(10, 6))
    # plt.hist(spray_base, bins=10, alpha=0.5, color='k')
    plt.hist(np.log10(-dev_spray), bins=10, alpha=0.5, color='r')
    plt.hist(np.log10(-dev_streak), bins=10, alpha=0.5, color='b')
    plt.hist(np.log10(-dev_first), bins=10, alpha=0.5, color='g')

    # plt.axvline(np.mean(spray_base), color='k', linestyle='--', label=f'Baseline: {np.mean(spray_base):.2f}')
    plt.axvline(np.log10(np.mean(-dev_spray)), color='r', linestyle='--', label=f'Spray: {np.mean(-dev_spray):.2f}')
    plt.axvline(np.log10(np.mean(-dev_streak)), color='b', linestyle='--', label=f'Streak: {np.mean(-dev_streak):.2f}')
    plt.axvline(np.log10(np.mean(-dev_first)), color='g', linestyle='--', label=f'1st order: {np.mean(-dev_first):.2f}')
    plt.legend(loc='best')
    plt.xlabel('Delta Chi2')
    plt.ylabel('Count')
    plt.savefig('./MockStreams/delta_chi2_comparison.pdf')
    plt.close()