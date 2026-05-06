"""JAX-compatible wrappers for Clapeyron.jl."""

from enum import StrEnum, auto

import jax.numpy as jnp
from jaxtyping import Array, Float

from .wrapper import create_jax_wrapper


# Enums
class Phase(StrEnum):
    """Phases accepted by Clapeyron.jl."""

    Unknown = auto()
    Liquid = auto()
    Vapor = auto()
    Solid = auto()
    Stable = auto()


# Utilities
def mass_to_moles(model, ws: Float[Array, "n"]) -> Float[Array, "n"]:
    """Convert a mass fraction into a mole fraction."""
    mw = 1e-3 * jnp.array(model.params.Mw.values)
    num = ws / mw
    return num / jnp.sum(num)


def moles_to_mass(model, zs: Float[Array, "n"]) -> Float[Array, "n"]:
    """Convert a mole fraction into a mass fraction."""
    mw = 1e-3 * jnp.array(model.params.Mw.values)
    num = zs * mw
    return num / jnp.sum(num)


# Pressure-based bulk properties
mass_density = create_jax_wrapper("mass_density", "(),(),(n)->()")

# Volume-based bulk properties
pressure = create_jax_wrapper("pressure", "(),(),(n)->()")

# Multiphase properties
bubble_pressure = create_jax_wrapper("bubble_pressure", "(),(n)->(),(),(),(n)")
saturation_pressure = create_jax_wrapper("saturation_pressure", "(),()->(),(),()")
