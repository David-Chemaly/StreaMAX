import jax
import jax.numpy as jnp
import jax.random as random

@jax.jit
def sample_params(seed):
    # seed = np.random.randint(0, 2**32 - 1)  # Ensure it's within JAX's valid range

    key = random.PRNGKey(seed)  # Set seed for reproducibility
    # Split key once for all parameters
    keys = random.split(key, 10)  # Generate enough subkeys at once

    # Generate random variables
    logM = random.uniform(keys[0], shape=(), minval=11, maxval=14)
    Rs   = random.uniform(keys[1], shape=(), minval=5, maxval=25)
    dirx, diry, dirz = random.normal(keys[3], shape=(3,))    # Mean 0, Std 1
    dirz = jnp.abs(dirz)  # Ensure positive direction

    logm = random.uniform(keys[4], shape=(), minval=7, maxval=9)
    rs   = random.uniform(keys[5], shape=(), minval=1, maxval=3)

    # Generate normal-distributed variables
    x0, z0 = random.normal(keys[6], shape=(2,)) * 150     # Mean 0, Std 50
    x0 = jnp.abs(x0)  # Ensure positive position
    z0 = jnp.abs(z0)
    y0 = 0. # Set to 0

    vx0, vy0, vz0 = random.normal(keys[7], shape=(3,)) * 250  # Mean 0, Std 50
    vy0 = jnp.abs(vy0)

    # Generate time
    time  = random.uniform(keys[8], shape=(), minval=1, maxval=4)
    alpha = random.uniform(keys[9], shape=(), minval=0.9, maxval=1.1)

    return logM, Rs, dirx, diry, dirz, logm, rs, x0, y0, z0, vx0, vy0, vz0, time, alpha

@jax.jit
def prior_transform(p):
    #ndim = 14
    logM, Rs, dirx, diry, dirz, \
    logm, rs, \
    x0, z0, vx0, vy0, vz0, \
    t0, a0 = p

    logM1 = 11 + 3 * logM
    Rs1   = 5 + 20 * Rs

    dirx1 = jax.scipy.special.ndtri(dirx)
    diry1 = jax.scipy.special.ndtri(diry)
    dirz1 = jax.scipy.special.ndtri(0.5 + dirz / 2)

    logm1 = 7 + 2 * logm
    rs1   = 1 + 2 * rs

    x1 = jax.scipy.special.ndtri(0.5 + x0 / 2) * 150
    z1 = jax.scipy.special.ndtri(0.5 + z0 / 2) * 150

    vx1 = jax.scipy.special.ndtri(vx0) * 250
    vy1 = jax.scipy.special.ndtri(0.5 + vy0 / 2) * 250
    vz1 = jax.scipy.special.ndtri(vz0) * 250

    t1 = 1 + 3 * t0
    a1 = 0.9 + 0.2 * a0
    
    return jnp.array([
        logM1, Rs1, dirx1, diry1, dirz1,
        logm1, rs1,
        x1, z1, vx1, vy1, vz1,
        t1, a1
    ])
