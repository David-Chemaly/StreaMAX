import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from flax import struct
import functools # Import functools for partial methods

from utils import get_mat

from constants import G, EPSILON, GYR_TO_S, KPC_TO_KM

### NFW Functions ###
@jax.jit
def NFWPotential(x, y, z, logM, Rs, q, dirx, diry, dirz):
    """
    Computes the NFW potential at a given position (x, y, z) with specified parameters.
    
    Args:
        x (float): x-coordinate.
        y (float): y-coordinate.
        z (float): z-coordinate.
        logM (float): Logarithm of the mass of the halo.
        Rs (float): Scale radius of the halo.
        q (float): Axis ratio.
        dirx (float): x-component of the direction vector.
        diry (float): y-component of the direction vector.
        dirz (float): z-component of the direction vector.

    Returns:
        float: The computed potential at the given position.
    """
    r_input = jnp.array([x, y, z])
    
    # Get rotation matrix
    rot_mat = get_mat(dirx, diry, dirz)
    
    r_vect = jnp.dot(rot_mat, r_input)
    rx, ry, rz = r_vect[0], r_vect[1], r_vect[2]
    
    r = jnp.sqrt(rx**2 + ry**2 + (rz / q)**2 + EPSILON)
    
    phi = -G * 10**logM / r * jnp.log(1 + r / Rs)
    return phi  # km²/s²

@jax.jit
def NFWAcceleration(x, y, z, logM, Rs, q, dirx, diry, dirz):
    """
    Computes the acceleration as the negative gradient of the NFW potential.
    
    Args:
        x (float): x-coordinate.
        y (float): y-coordinate.
        z (float): z-coordinate.
        logM (float): Logarithm of the mass of the halo.
        Rs (float): Scale radius of the halo.
        q (float): Axis ratio.
        dirx (float): x-component of the direction vector.
        diry (float): y-component of the direction vector.
        dirz (float): z-component of the direction vector.

    Returns:
        jnp.ndarray: The acceleration vector at the given position.
    """
    potential_func = lambda pos: NFWPotential(pos[0], pos[1], pos[2], logM, Rs, q, dirx, diry, dirz)
    
    acc = jax.grad(potential_func)(jnp.array([x, y, z]))
    return -acc * GYR_TO_S  # km² / s / Gyr / kpc

@jax.jit
def NFWHessian(x, y, z, logM, Rs, q, dirx, diry, dirz):
    """
    Computes the Hessian matrix of the NFW potential at a given position.
    
    Args:
        x (float): x-coordinate.
        y (float): y-coordinate.
        z (float): z-coordinate.
        logM (float): Logarithm of the mass of the halo.
        Rs (float): Scale radius of the halo.
        q (float): Axis ratio.
        dirx (float): x-component of the direction vector.
        diry (float): y-component of the direction vector.
        dirz (float): z-component of the direction vector.

    Returns:
        jnp.ndarray: The Hessian matrix at the given position.
    """
    potential_func = lambda pos: NFWPotential(pos[0], pos[1], pos[2], logM, Rs, q, dirx, diry, dirz)
    
    hess = jax.hessian(potential_func)(jnp.array([x, y, z]))
    return hess  # km² / s² / kpc²

### Plummer Functions ###
@jax.jit
def PlummerPotential(x, y, z, logM, Rs, x_origin=0.0, y_origin=0.0, z_origin=0.0):
    """
    Computes the Plummer potential at a given position (x, y, z) with specified parameters.
    
    Args:
        x (float): x-coordinate.
        y (float): y-coordinate.
        z (float): z-coordinate.
        logM (float): Logarithm of the mass of the halo.
        Rs (float): Scale radius of the halo.
        x_origin (float): x-coordinate of the origin.
        y_origin (float): y-coordinate of the origin.
        z_origin (float): z-coordinate of the origin.

    Returns:
        float: The computed potential at the given position.
    """
    r = jnp.sqrt((x - x_origin)**2 + (y - y_origin)**2 + (z - z_origin)**2 + EPSILON)
    phi = -G * 10**logM / jnp.sqrt(r**2 + Rs**2)
    return phi # km²/s²

@jax.jit
def PlummerAcceleration(x, y, z, logM, Rs, x_origin=0.0, y_origin=0.0, z_origin=0.0):
    """
    Computes the acceleration as the negative gradient of the Plummer potential.
    
    Args:
        x (float): x-coordinate.
        y (float): y-coordinate.
        z (float): z-coordinate.
        logM (float): Logarithm of the mass of the halo.
        Rs (float): Scale radius of the halo.
        x_origin (float): x-coordinate of the origin.
        y_origin (float): y-coordinate of the origin.
        z_origin (float): z-coordinate of the origin.

    Returns:
        jnp.ndarray: The acceleration vector at the given position.
    """
    potential_func = lambda pos: PlummerPotential(pos[0], pos[1], pos[2], logM, Rs, x_origin, y_origin, z_origin)
    
    acc = jax.grad(potential_func)(jnp.array([x, y, z]))
    return -acc * GYR_TO_S # km² / s / Gyr / kpc

@jax.jit
def PlummerHessian(x, y, z, logM, Rs, x_origin= 0.0, y_origin=0.0, z_origin=0.0):
    """
    Computes the Hessian matrix of the Plummer potential at a given position.
    
    Args:
        x (float): x-coordinate.
        y (float): y-coordinate.
        z (float): z-coordinate.
        logM (float): Logarithm of the mass of the halo.
        Rs (float): Scale radius of the halo.
        x_origin (float): x-coordinate of the origin.
        y_origin (float): y-coordinate of the origin.
        z_origin (float): z-coordinate of the origin.

    Returns:
        jnp.ndarray: The Hessian matrix at the given position.
    """
    potential_func = lambda pos: PlummerPotential(pos[0], pos[1], pos[2], logM, Rs, x_origin, y_origin, z_origin)
    
    hess = jax.hessian(potential_func)(jnp.array([x, y, z]))
    return hess # km² / s² / kpc²