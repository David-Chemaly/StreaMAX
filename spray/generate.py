import sys
sys.path.append('..')
from constants import G
from potentials import NFWAcceleration, PlummerAcceleration
from integrants import integrate_satellite


if __name__ == "__main__":
    # Example parameters
    logM, Rs, q = 12.0, 10.0, 0.8
    dirx, diry, dirz = 1.0, 1.0, 0.0
    x0, y0, z0 = 100.0, 0.0, 0.0
    vx0, vy0, vz0 = 0.0, 100.0, 0.0
    time = 2.0

    # Call the function
    trajectory, time_steps = integrate_satellite(x0, y0, z0, vx0, vy0, vz0, logM, Rs, q, dirx, diry, dirz, time, N_STEPS=100)
    trajectory, time_steps = integrate_satellite(x0, y0, z0, vx0, vy0, vz0, logM, Rs, q, dirx, diry, dirz, time, N_STEPS=200)