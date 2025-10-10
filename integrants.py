import jax
import jax.numpy as jnp
from functools import partial

# ---------- helpers ----------
def _split(w):
    return w[:3], w[3:]

def _merge(r, v):
    return jnp.concatenate([r, v], axis=0)

### Satellite Functions ###
@jax.jit
def leapfrog_satellite_step(state, dt, logM, Rs, q, dirx, diry, dirz):
    """
    Leapfrog integration step for satellite motion for NFW potential.
    """
    x, y, z, vx, vy, vz = state

    ax, ay, az = NFWAcceleration(x, y, z, logM, Rs, q, dirx, diry, dirz)

    vx_half = vx + 0.5 * dt * ax
    vy_half = vy + 0.5 * dt * ay
    vz_half = vz + 0.5 * dt * az

    x_new = x + dt * vx_half
    y_new = y + dt * vy_half
    z_new = z + dt * vz_half

    ax_new, ay_new, az_new = NFWAcceleration(x_new, y_new, z_new, logM, Rs, q, dirx, diry, dirz)

    vx_new = vx_half + 0.5 * dt * ax_new
    vy_new = vy_half + 0.5 * dt * ay_new
    vz_new = vz_half + 0.5 * dt * az_new

    return (x_new, y_new, z_new, vx_new, vy_new, vz_new)

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
    """Leapfrog (KDK) — returns full time grid and trajectory (n_steps+1,6)."""

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