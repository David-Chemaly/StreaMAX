import numpy as np

from utils import get_q, get_track
from spray import generate_stream_spray

BAD_VAL = -1e100

def log_likelihood(params, dict_data, seed=111, min_count=100):
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