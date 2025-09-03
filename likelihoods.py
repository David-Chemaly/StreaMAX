import os
import pickle
from tqdm import tqdm
import numpy as np

from utils import get_q, get_track, inference_first
from spray import generate_stream_spray
from first import generate_stream_first

BAD_VAL = -1e100

def log_likelihood(params, dict_data, seed=13, min_count=100):
    q      = get_q(params[2], params[3], params[4])
    params = np.concatenate([params[:2], [q], params[2:8], [0.], params[8:], [1.]])

    theta_stream, xv_stream, _, _ = generate_stream_spray(params,  seed)

    # theta_stream_first, xv_stream_first, _, _, S, _, refs, _ = generate_stream_first(params,  seed=seed)
    # theta_stream, xv_stream = inference_first(theta_stream_first, xv_stream_first, refs, S, seed=seed, disp_x=0.1, disp_v=1.)

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
    sigma = 1

    logl = []
    for seed in seeds:
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
        logl.append(log_likelihood(params_data, dict_data, seed=seed+1))
    
    print(np.mean(logl))