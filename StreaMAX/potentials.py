import jax
import jax.numpy as jnp

from .utils import get_mat

from .constants import G, EPSILON

# ---------- helpers ----------
@jax.jit
def _shift(x, y, z, p):
    return jnp.array([x - p['x_origin'], y - p['y_origin'], z - p['z_origin']])

@jax.jit
def _rotate(vec, p):
    R = get_mat(p['dirx'], p['diry'], p['dirz'])
    return R @ vec  # matvec

# ---------- Point Mass ----------
# Phi = - G M / r
@jax.jit
def PointMass_potential(x, y, z, params):
    '''
    params: dict with keys 'logM', 'x_origin', 'y_origin', 'z_origin', 'dirx', 'diry', 'dirz'
    '''
    r  = _shift(x, y, z, params)
    s  = jnp.sqrt(r @ r + EPSILON)
    return -G * 10**params['logM'] / s # kpc^2 / Gyr^2

@jax.jit
def PointMass_acceleration(x, y, z, params):
    def potential_vec(pos):
        return PointMass_potential(pos[0], pos[1], pos[2], params)
    grad_phi = jax.grad(potential_vec)(jnp.array([x, y, z]))
    return -grad_phi

@jax.jit
def PointMass_hessian(x, y, z, params):
    def potential_vec(pos):
        return PointMass_potential(pos[0], pos[1], pos[2], params)
    hess_phi = jax.hessian(potential_vec)(jnp.array([x, y, z]))
    return hess_phi

# ---------- Isochrone ----------
# Phi = - G M / ( b + sqrt(r^2 + b^2) )
@jax.jit
def Isochrone_potential(x, y, z, params):
    '''
    params: dict with keys 'logM', 'Rs', 'x_origin', 'y_origin', 'z_origin', 'dirx', 'diry', 'dirz'
    '''
    r = _shift(x, y, z, params)
    s  = jnp.sqrt(r @ r + params['Rs']*params['Rs'] + EPSILON)
    return -G * 10**params['logM'] / (params['Rs'] + s)  # kpc^2 / Gyr^2

@jax.jit
def Isochrone_acceleration(x, y, z, params):
    def potential_vec(pos):
        return Isochrone_potential(pos[0], pos[1], pos[2], params)
    grad_phi = jax.grad(potential_vec)(jnp.array([x, y, z]))
    return -grad_phi

@jax.jit
def Isochrone_hessian(x, y, z, params):
    def potential_vec(pos):
        return Isochrone_potential(pos[0], pos[1], pos[2], params)
    hess_phi = jax.hessian(potential_vec)(jnp.array([x, y, z]))
    return hess_phi

# ---------- Plummer ----------
# Phi = - G M / sqrt(r^2 + Rs^2)
@jax.jit
def Plummer_potential(x, y, z, params):
    '''
    params: dict with keys 'logM', 'Rs', 'x_origin', 'y_origin', 'z_origin', 'dirx', 'diry', 'dirz'
    '''
    r = _shift(x, y, z, params)
    s = jnp.sqrt(r @ r + params['Rs']*params['Rs'] + EPSILON)
    return -G * 10**params['logM'] / s  # kpc^2 / Gyr^2

@jax.jit
def Plummer_acceleration(x, y, z, params):
    def potential_vec(pos):
        return Plummer_potential(pos[0], pos[1], pos[2], params)
    grad_phi = jax.grad(potential_vec)(jnp.array([x, y, z]))
    return -grad_phi

@jax.jit
def Plummer_hessian(x, y, z, params):
    def potential_vec(pos):
        return Plummer_potential(pos[0], pos[1], pos[2], params)
    hess_phi = jax.hessian(potential_vec)(jnp.array([x, y, z]))
    return hess_phi

# ---------- (Triaxial) NFW ----------
# Phi = - G M / r * log(1 + r/Rs),  r = sqrt((rx/a)^2 + (ry/b)^2 + (rz/c)^2)
@jax.jit
def NFW_potential(x, y, z, params):
    '''
    params: dict with keys 'logM', 'Rs', 'a', 'b', 'c', 'x_origin', 'y_origin', 'z_origin', 'dirx', 'diry', 'dirz'
    '''
    rin = _shift(x, y, z, params)
    rvec = _rotate(rin, params)  
    rx, ry, rz = rvec
    r = jnp.sqrt((rx/params['a'])**2 + (ry/params['b'])**2 + (rz/params['c'])**2 + EPSILON)
    return -G * 10**params['logM'] * jnp.log(1 + r / params['Rs']) / (r + EPSILON)  # kpc^2 / Gyr^2

@jax.jit
def NFW_acceleration(x, y, z, params):
    def potential_vec(pos):
        return NFW_potential(pos[0], pos[1], pos[2], params)
    grad_phi = jax.grad(potential_vec)(jnp.array([x, y, z]))
    return -grad_phi

@jax.jit
def NFW_hessian(x, y, z, params):
    def potential_vec(pos):
        return NFW_potential(pos[0], pos[1], pos[2], params)
    hess_phi = jax.hessian(potential_vec)(jnp.array([x, y, z]))
    return hess_phi

# ---------- Miyamoto-Nagai Disk ----------
# Phi = - G M / sqrt(R^2 + (Rs + sqrt(z^2 + Hs^2))^2)
@jax.jit
def MiyamotoNagai_potential(x, y, z, params):
    '''
    params: dict with keys 'logM', 'Rs', 'Hs', 'x_origin', 'y_origin', 'z_origin', 'dirx', 'diry', 'dirz'
    '''
    rin  = _shift(x, y, z, params)
    rvec = _rotate(rin, params)  
    rx, ry, rz = rvec + EPSILON

    R = (rx**2 + ry**2)**0.5

    denom2 = (R**2 + (params['Rs'] + (rz**2 + params['Hs']**2)**0.5)**2)

    Phi = - G * 10**params['logM'] / (denom2**0.5)

    return Phi

@jax.jit
def MiyamotoNagai_acceleration(x, y, z, params):
    def potential_vec(pos):
        return MiyamotoNagai_potential(pos[0], pos[1], pos[2], params)
    grad_phi = jax.grad(potential_vec)(jnp.array([x, y, z]))
    return -grad_phi

@jax.jit
def MiyamotoNagai_hessian(x, y, z, params):
    def potential_vec(pos):
        return MiyamotoNagai_potential(pos[0], pos[1], pos[2], params)
    hess_phi = jax.hessian(potential_vec)(jnp.array([x, y, z]))
    return hess_phi

# ---------- Quadratic Bar ----------
# Phi = A * Rs^3 * (R^2 / (Rs + R)^5) * cos(2*phi_bar) * exp(-|z|/Hs)
# where phi_bar = phi - Omega*(t - t0)
# and phi = arctan(y/x)
# and eps = 2*(t-t0)/(t1-t0) - 1
# and val = 0 if t<t0, 1 if t>t1, and 3/16*eps^5 - 5/8*eps^3 + 15/16*eps + 1/2 if t0<=t<=t1
@jax.jit
def Bar_potential(x, y, z, t, params):
    '''
    params: dict with keys 'A', 'Rs', 'Hs', 'Omega', 't0', 't1', 'x_origin', 'y_origin', 'z_origin', 'dirx', 'diry', 'dirz'
    '''
    rin  = _shift(x, y, z, params)
    rvec = _rotate(rin, params)
    rx, ry, rz = rvec + EPSILON

    R   = jnp.sqrt(rx**2 + ry**2)
    phi = jnp.arctan2(ry, rx)

    phi_bar = phi - params['Omega'] * (t - params['t0'])

    eps = 2.0*(t - params['t0'])/(params['t1'] - params['t0']) - 1.0
    val = jax.lax.cond(
        t < params['t0'],
        lambda _: 0.0,
        lambda _: jax.lax.cond(
            t > params['t1'],
            lambda _: 1.0,
            lambda _: (3.0/16.0)*eps**5 - (5.0/8.0)*eps**3 + (15.0/16.0)*eps + 0.5,
            operand=None
        ),
        operand=None
    )

    amp = params['A'] * params['Rs']**3 * (R**2 / (params['Rs'] + R)**5) * val

    return amp * jnp.cos(2*phi_bar) * (jnp.exp(-(rz/params['Hs'])**2))  # kpc^2 / Gyr^2

@jax.jit
def Bar_acceleration(x, y, z, t, params):
    def potential_vec(pos):
        return Bar_potential(pos[0], pos[1], pos[2], t, params)
    grad_phi = jax.grad(potential_vec)(jnp.array([x, y, z]))
    return -grad_phi

@jax.jit
def Bar_hessian(x, y, z, t, params):
    def potential_vec(pos):
        return Bar_potential(pos[0], pos[1], pos[2], t, params)
    hess_phi = jax.hessian(potential_vec)(jnp.array([x, y, z]))
    return hess_phi

