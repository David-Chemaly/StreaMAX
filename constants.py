from astropy import units as auni
from astropy.constants import G
from jax import numpy as jnp

G = G.to(auni.kpc**3/auni.Msun/auni.Gyr**2).value # kpc3 / Msun / Gyr2

KPC_TO_KM     = (1 * auni.kpc/auni.km).to(auni.km/auni.km).value
GYR_TO_S      = (1 * auni.Gyr/auni.s).to(auni.s/auni.s).value
KMS_TO_KPCGYR = GYR_TO_S / KPC_TO_KM
KPCGYR_TO_KMS = KPC_TO_KM / GYR_TO_S
EPSILON       = 1e-12  # Small constant to avoid division by zero
TWOPI         = 2 * jnp.pi