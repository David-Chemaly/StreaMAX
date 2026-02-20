"""
StreaMAX — a JAX-accelerated stellar stream generator and analysis toolkit.
"""

__version__ = "0.1.0"

# Enforce float32 on both CPU and GPU (must be set before any JAX operations)
import jax
jax.config.update("jax_enable_x64", False)

# Optional: expose key functions at the top level
from .generator import *
from .potentials import *
from .utils import *
from .constants import *
from .integrants import *
from .methods import *