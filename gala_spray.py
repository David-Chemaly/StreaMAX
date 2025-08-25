import numpy as np
import gala.potential as gp
import gala.dynamics as gd
from gala.dynamics import mockstream as ms


from utils import get_mat
from astropy import units as auni

def model_stream(params, dt=-10):
    logM, Rs, q, dirx, diry, dirz, \
    logm, rs, \
    pos_init_x, pos_init_y, pos_init_z, \
    vel_init_x, vel_init_y, vel_init_z, \
    t_end = params

    units = [auni.kpc, auni.km / auni.s, auni.Msun, auni.Gyr, auni.rad]

    w0 = gd.PhaseSpacePosition(
        pos=np.array([pos_init_x, pos_init_y, pos_init_z]) * auni.kpc,
        vel=np.array([vel_init_x, vel_init_y, vel_init_z]) * auni.km / auni.s,
    )

    mat = get_mat(dirx, diry, dirz)

    pot = gp.NFWPotential(10**logM, Rs, 1, 1, q, R=mat, units=units)

    H = gp.Hamiltonian(pot)

    df = ms.FardalStreamDF(gala_modified=True, random_state=np.random.RandomState(42))

    prog_pot = gp.PlummerPotential(m=10**logm, b=rs, units=units)
    gen = ms.MockStreamGenerator(df, H, progenitor_potential=prog_pot)

    stream, _ = gen.run(w0, 10**logm * auni.Msun, dt=dt* auni.Myr, n_steps=int(t_end * auni.Gyr/ abs(dt* auni.Myr)))
    xy_stream = stream.xyz.T[:, :2]

    return xy_stream.value