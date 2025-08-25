import agama, numpy, matplotlib, matplotlib.pyplot as plt, time

numpy.set_printoptions(precision=6, linewidth=999, suppress=True)
numpy.random.seed(111)
plt.rc('font', size=12)
plt.rc('figure', dpi=75)
plt.rc('mathtext', fontset='stix')
# all dimensional quantities are given in the units of 1 kpc, 1 km/s, 1 Msun;
agama.setUnits(length=1, velocity=1, mass=1)
timeUnitGyr = agama.getUnits()['time'] / 1e3  # time unit is 1 kpc / (1 km/s)
print('time unit: %.3f Gyr' % timeUnitGyr)

def get_rj_vj_R(pot_host, orbit_sat, mass_sat):
    """
    Compute the Jacobi radius, associated velocity, and rotation matrix
    for generating streams using particle-spray methods.
    Arguments:
        pot_host:  an instance of agama.Potential for the host galaxy.
        orbit_sat: the orbit of the satellite, an array of shape (N, 6).
        mass_sat:  the satellite mass (a single number or an array of length N).
    Return:
        rj:  Jacobi radius at each point on the orbit (length: N).
        vj:  velocity offset from the satellite at each point on the orbit (length: N).
        R:   rotation matrix converting from host to satellite at each point on the orbit (shape: N,3,3)
    """
    N = len(orbit_sat)
    x, y, z, vx, vy, vz = orbit_sat.T
    Lx = y * vz - z * vy
    Ly = z * vx - x * vz
    Lz = x * vy - y * vx
    r = (x*x + y*y + z*z)**0.5
    L = (Lx*Lx + Ly*Ly + Lz*Lz)**0.5
    # rotation matrices transforming from the host to the satellite frame for each point on the trajectory
    R = numpy.zeros((N, 3, 3))
    R[:,0,0] = x/r
    R[:,0,1] = y/r
    R[:,0,2] = z/r
    R[:,2,0] = Lx/L
    R[:,2,1] = Ly/L
    R[:,2,2] = Lz/L
    R[:,1,0] = R[:,0,2] * R[:,2,1] - R[:,0,1] * R[:,2,2]
    R[:,1,1] = R[:,0,0] * R[:,2,2] - R[:,0,2] * R[:,2,0]
    R[:,1,2] = R[:,0,1] * R[:,2,0] - R[:,0,0] * R[:,2,1]
    # compute  the second derivative of potential by spherical radius
    der = pot_host.eval(orbit_sat[:,0:3], der=True)
    d2Phi_dr2 = -(x**2  * der[:,0] + y**2  * der[:,1] + z**2  * der[:,2] +
                  2*x*y * der[:,3] + 2*y*z * der[:,4] + 2*z*x * der[:,5]) / r**2
    # compute the Jacobi radius and the relative velocity at this radius for each point on the trajectory
    Omega = L / r**2
    rj = (agama.G * mass_sat / (Omega**2 - d2Phi_dr2))**(1./3)
    vj = Omega * rj
    return rj, vj, R

def create_ic_particle_spray(orbit_sat, rj, vj, R, gala_modified=True):
    """
    Create initial conditions for particles escaping through Largange points,
    using the method of Fardal+2015
    Arguments:
        orbit_sat:  the orbit of the satellite, an array of shape (N, 6).
        rj:  Jacobi radius at each point on the orbit (length: N).
        vj:  velocity offset from the satellite at each point on the orbit (length: N).
        R:   rotation matrix converting from host to satellite at each point on the orbit (shape: N,3,3)
        gala_modified:  if True, use modified parameters as in Gala, otherwise the ones from the original paper.
    Return:
        initial conditions for stream particles, an array of shape (2*N, 6) - 
        two points for each point on the original satellite trajectory.
    """
    N = len(rj)
    # assign positions and velocities (in the satellite reference frame) of particles
    # leaving the satellite at both lagrange points (interleaving positive and negative offsets).
    rj = numpy.repeat(rj, 2) * numpy.tile([1, -1], N)
    vj = numpy.repeat(vj, 2) * numpy.tile([1, -1], N)
    R  = numpy.repeat(R, 2, axis=0)
    mean_x  = 2.0
    disp_x  = 0.5 if gala_modified else 0.4
    disp_z  = 0.5
    mean_vy = 0.3
    disp_vy = 0.5 if gala_modified else 0.4
    disp_vz = 0.5
    rx  = numpy.random.normal(size=2*N) * disp_x + mean_x
    rz  = numpy.random.normal(size=2*N) * disp_z * rj
    rvy =(numpy.random.normal(size=2*N) * disp_vy + mean_vy) * vj * (rx if gala_modified else 1)
    rvz = numpy.random.normal(size=2*N) * disp_vz * vj
    rx *= rj
    offset_pos = numpy.column_stack([rx,  rx*0, rz ])  # position and velocity of particles in the reference frame
    offset_vel = numpy.column_stack([rx*0, rvy, rvz])  # centered on the progenitor and aligned with its orbit
    ic_stream = numpy.tile(orbit_sat, 2).reshape(2*N, 6)   # same but in the host-centered frame
    ic_stream[:,0:3] += numpy.einsum('ni,nij->nj', offset_pos, R)
    ic_stream[:,3:6] += numpy.einsum('ni,nij->nj', offset_vel, R)
    return ic_stream

def create_stream_particle_spray(time_total, num_particles, pot_host, posvel_sat, mass_sat, gala_modified=True):
    """
    Construct a stream using the particle-spray method.
    Arguments:
        time_total:  duration of time for stream generation 
            (positive; orbit of the progenitor integrated from present day (t=0) back to time -time_total).
        num_particles:  number of points in the stream (even; divided equally between leading and trailing arms).
        pot_host:    an instance of agama.Potential for the host galaxy.
        posvel_sat:  present-day position and velocity of the satellite (array of length 6).
        mass_sat:    the satellite mass (a single number or an array of length num_particles//2).
        gala_modified:  if True, use modified parameters as in Gala, otherwise the ones from the original paper.
    Return:
        xv_stream: position and velocity of stream particles at present time (shape: num_particles, 6).
    """
    # number of points on the orbit: each point produces two stream particles (leading and trailing arms)
    N = num_particles//2

    # integrate the orbit of the progenitor from its present-day posvel (at time t=0)
    # back in time for an interval time_total, storing the trajectory at N points
    time_sat, orbit_sat = agama.orbit(
        potential=pot_host, ic=posvel_sat, time=-time_total, trajsize=N+1)
    # remove the 0th point (the present-day posvel) and reverse the arrays to make them increasing in time
    time_sat  = time_sat [1:][::-1]
    orbit_sat = orbit_sat[1:][::-1]

    # at each point on the trajectory, create a pair of seed initial conditions
    # for particles released at both Lagrange points
    rj, vj, R = get_rj_vj_R(pot_host, orbit_sat, mass_sat)
    ic_stream = create_ic_particle_spray(orbit_sat, rj, vj, R, gala_modified)
    time_seed = numpy.repeat(time_sat, 2)
    result = agama.orbit(
        potential=pot_host, ic=ic_stream,
        timestart=time_seed,  # starting time for each orbit (negative)
        time=-time_seed,      # integration time for each orbit: time_start + time = 0 (end time, i.e. now)
        trajsize=1)           # each orbit produces just one point (at the end of the integration)
    # 0th column is the array of times (here, just one element for each orbit, i.e. its end time),
    # 1st column is the trajectory (here, one end point per orbit)
    xv_stream = numpy.vstack(result[:,1])
    
    return xv_stream