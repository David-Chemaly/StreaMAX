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
@partial(jax.jit, static_argnames=('acc_fn_host', 'acc_fn_prog', 'n_steps', 'unroll'))
def combined_integrate_leapfrog_final(index, w0, params_host, 
                                        prog_traj, params_prog, 
                                        acc_fn_host, acc_fn_prog, 
                                        n_steps, dt, 
                                        prog_mass, dm, 
                                        unroll=True):
    dt = dt[index]
    dm = dm[index]

    def step(carry, _):
        t, y, z, g, m = carry
        r, v = _split(y)
        rp, vp = _split(z)
        
        params_prog['logM'] = m
        params_prog['x_origin'] = rp[0]
        params_prog['y_origin'] = rp[1]
        params_prog['z_origin'] = rp[2]

        a0      = acc_fn_host(*r, params_host) + acc_fn_prog(*r, params_prog)
        a0_prog = acc_fn_host(*rp, params_host)

        v_half = v + 0.5 * dt * a0
        v_half_prog = vp + 0.5 * dt * a0_prog

        r_new = r + dt * v_half
        r_new_prog = rp + dt * v_half_prog

        m_new = m - dm
        params_prog['logM'] = m_new
        params_prog['x_origin'] = r_new_prog[0]
        params_prog['y_origin'] = r_new_prog[1]
        params_prog['z_origin'] = r_new_prog[2]

        a1      = acc_fn_host(*r_new, params_host) + acc_fn_prog(*r_new, params_prog)
        a1_prog = acc_fn_host(*r_new_prog, params_host)

        v_new = v_half + 0.5 * dt * a1
        v_new_prog = v_half_prog + 0.5 * dt * a1_prog

        y_new = _merge(r_new, v_new)
        z_new = _merge(r_new_prog, v_new_prog)
        t_new = t + dt

        g_new = g + jnp.sum(rp**2)**0.5 - jnp.sum(r_new**2)**0.5
        return (t_new, y_new, z_new, g_new, m_new), None

    t0 = 0.
    g0 = 0.
    p0 = prog_traj[index]
    m0 = prog_mass[index]
    (_, wN, _, gN, _), _ = jax.lax.scan(step, (t0, w0, p0, g0, m0), xs=None, length=n_steps, unroll=unroll)
    return wN, gN