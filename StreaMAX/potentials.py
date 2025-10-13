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
    b = params['Rs']
    r = _shift(x, y, z, params)
    s  = jnp.sqrt(r @ r + b*b + EPSILON)
    return -G * 10**params['logM'] / (b + s)  # kpc^2 / Gyr^2

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
# Phi = - G M / sqrt(r^2 + b^2)
@jax.jit
def Plummer_potential(x, y, z, params):
    b = params['Rs']
    r = _shift(x, y, z, params)
    s = jnp.sqrt(r @ r + b*b + EPSILON)
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
    Rs = params['Rs']
    ax = params['a']
    by = params['b']
    cz = params['c']
    rin = _shift(x, y, z, params)
    rvec = _rotate(rin, params)  
    rx, ry, rz = rvec
    r = jnp.sqrt((rx/ax)**2 + (ry/by)**2 + (rz/cz)**2 + EPSILON)
    return -G * 10**params['logM'] * jnp.log(1 + r / Rs) / (r + EPSILON)  # kpc^2 / Gyr^2

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