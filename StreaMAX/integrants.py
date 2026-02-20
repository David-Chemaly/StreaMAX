import jax
import jax.numpy as jnp
from functools import partial

# ---------- helpers ----------
def _split(w):
    return w[:3], w[3:]

def _merge(r, v):
    return jnp.concatenate([r, v], axis=0)

# ---------- integrators ----------
@partial(jax.jit, static_argnames=('acc_fn', 'n_steps', 'unroll'))
def integrate_leapfrog_final(w0, params, acc_fn, n_steps, dt = 0.010, t0 = 0.0, unroll=True):
    """Leapfrog (KDK) — returns final time and final state only."""

    def step(carry, _):
        t, y = carry
        r, v = _split(y)
        a0 = acc_fn(*r, params)
        v_half = v + 0.5 * dt * a0
        r_new = r + dt * v_half
        a1 = acc_fn(*r_new, params)
        v_new = v_half + 0.5 * dt * a1
        y_new = _merge(r_new, v_new)
        t_new = t + dt
        return (t_new, y_new), None

    (tN, wN), _ = jax.lax.scan(step, (t0, w0), xs=None, length=n_steps, unroll=unroll)
    return tN, wN

@partial(jax.jit, static_argnames=('acc_fn', 'n_steps', 'unroll'))
def integrate_leapfrog_traj(w0, params, acc_fn, n_steps, dt = 0.010, t0 = 0.0, unroll=True):
    """Leapfrog (KDK) — returns full time grid and trajectory."""

    def step(y, _):
        r, v = _split(y)
        a0 = acc_fn(*r, params)
        v_half = v + 0.5 * dt * a0
        r_new = r + dt * v_half
        a1 = acc_fn(*r_new, params)
        v_new = v_half + 0.5 * dt * a1
        y_new = _merge(r_new, v_new)
        return y_new, y_new

    _, Ys = jax.lax.scan(step, w0, xs=None, length=n_steps, unroll=unroll)
    Y = jnp.vstack([w0, Ys])
    ts = t0 + dt * jnp.arange(n_steps + 1, dtype=w0.dtype)
    return ts, Y

# ---------- combined potential integrators ----------

@partial(jax.jit, static_argnames=('acc_fn', 'n_steps', 'unroll'))
def precompute_prog_trajectories(xv_sat, params_host, acc_fn, n_steps, dt_arr, m_sat, dm_arr, unroll=True):
    """For each orbit release step i, integrate the progenitor forward n_steps steps.

    Returns:
        prog_pos : (n_steps+1, n_steps+1, 3)  — progenitor position at each step boundary
        prog_M   : (n_steps+1, n_steps+1)     — linear mass 10^logM at each step boundary
    """
    def integrate_one(w0, dt, m0, dm):
        def step(carry, _):
            pos_vel, m = carry
            r, v = pos_vel[:3], pos_vel[3:]
            a0    = acc_fn(*r, params_host)
            v_half = v + 0.5 * dt * a0
            r_new  = r + dt * v_half
            a1    = acc_fn(*r_new, params_host)
            v_new  = v_half + 0.5 * dt * a1
            m_new  = m - dm
            return (jnp.concatenate([r_new, v_new]), m_new), (r_new, m_new)

        _, (positions, masses) = jax.lax.scan(
            step, (w0, m0), xs=None, length=n_steps, unroll=unroll
        )
        # prepend boundary 0 (the starting state)
        all_pos = jnp.concatenate([w0[:3][None, :], positions], axis=0)  # (n_steps+1, 3)
        all_m   = jnp.concatenate([m0[None], masses],           axis=0)  # (n_steps+1,)
        return all_pos, all_m

    # vmap over all n_steps+1 orbit release points
    prog_pos, prog_logM = jax.vmap(integrate_one)(xv_sat, dt_arr, m_sat, dm_arr)
    # prog_pos   : (n_steps+1, n_steps+1, 3)
    # prog_logM  : (n_steps+1, n_steps+1)
    return prog_pos, 10.**prog_logM  # return linear mass directly


@partial(jax.jit, static_argnames=('acc_fn_host', 'acc_fn_prog', 'n_steps', 'unroll'))
def combined_integrate_leapfrog_final(index, w0, params_host,
                                        prog_pos, prog_M,
                                        params_prog,
                                        acc_fn_host, acc_fn_prog,
                                        n_steps, dt,
                                        unroll=True):
    """Integrate one stream particle, looking up progenitor state from precomputed tables.

    prog_pos : (n_steps+1, n_steps+1, 3)  — precomputed progenitor positions
    prog_M   : (n_steps+1, n_steps+1)     — precomputed linear progenitor masses
    """
    dt = dt[index]

    def step(carry, step_s):
        y, g = carry
        r, v = _split(y)

        # Look up progenitor state at the start and end of this leapfrog step
        rp   = prog_pos[index, step_s]       # progenitor position at boundary s
        rp_n = prog_pos[index, step_s + 1]   # progenitor position at boundary s+1
        M    = prog_M[index, step_s]          # linear mass at boundary s
        M_n  = prog_M[index, step_s + 1]     # linear mass at boundary s+1

        pp0 = {**params_prog, 'M': M,
               'x_origin': rp[0],   'y_origin': rp[1],   'z_origin': rp[2]}
        a0  = acc_fn_host(*r, params_host) + acc_fn_prog(*r, pp0)

        v_half = v + 0.5 * dt * a0
        r_new  = r + dt * v_half

        pp1 = {**params_prog, 'M': M_n,
               'x_origin': rp_n[0], 'y_origin': rp_n[1], 'z_origin': rp_n[2]}
        a1  = acc_fn_host(*r_new, params_host) + acc_fn_prog(*r_new, pp1)

        v_new = v_half + 0.5 * dt * a1
        y_new = _merge(r_new, v_new)

        g_new = g + jnp.linalg.norm(rp) - jnp.linalg.norm(r_new)
        return (y_new, g_new), None

    g0 = 0.
    (wN, gN), _ = jax.lax.scan(
        step, (w0, g0), xs=jnp.arange(n_steps), length=n_steps, unroll=unroll
    )
    return wN, gN