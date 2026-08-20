"""JAX-compatible wrappers for Clapeyron.jl."""

import importlib.metadata

import jax.numpy as jnp
from jaxtyping import Array, Float

from ._phases import (
    Liquid as Liquid,
    Phase as Phase,
    Solid as Solid,
    Stable as Stable,
    Unknown as Unknown,
    Vapor as Vapor,
)
from .wrapper import create_jax_wrapper

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0-dev"  # Fallback for editable installations


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
mass_internal_energy = create_jax_wrapper("mass_internal_energy", "(),(),(n)->()")

# Volume-based bulk properties
pressure = create_jax_wrapper("pressure", "(),(),(n)->()")

# Multiphase properties
bubble_pressure = create_jax_wrapper("bubble_pressure", "(),(n)->(),(),(),(n)")
saturation_pressure = create_jax_wrapper("saturation_pressure", "(),()->(),(),()")
