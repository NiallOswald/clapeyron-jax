"""JAX-compatible wrappers for Clapeyron.jl."""

import pyclapeyron as cl

from .wrapper import create_jax_wrapper

# Pressure-based bulk properties
mass_density = create_jax_wrapper(cl.mass_density, "(),(),(n)->()")

# Volume-based bulk properties
pressure = create_jax_wrapper(cl.pressure, "(),(),(n)->()")

# Multiphase properties
bubble_pressure = create_jax_wrapper(cl.bubble_pressure, "(),(),(n)->(),(),(),(n)")
