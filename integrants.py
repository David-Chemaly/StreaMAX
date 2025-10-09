import jax
import jax.numpy as jnp

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