"""JAX-compatible wrappers for Clapeyron.jl."""

import importlib.metadata

import jax.numpy as jnp
from jaxtyping import Array, Float

from . import VT0 as VT0
from ._phases import (
    Liquid as Liquid,
    Phase as Phase,
    Solid as Solid,
    Stable as Stable,
    Unknown as Unknown,
    Vapor as Vapor,
)
from ._wrapper import create_jax_wrapper

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0-dev"  # Fallback for editable installations


# Utilities
def mass_to_moles(model, ws: Float[Array, "n"]) -> Float[Array, "n"]:
    """Convert a mass fraction into a mole fraction."""
    mw = 1e-3 * jnp.array(model.params.Mw.values)
    moles = ws / mw
    return moles / jnp.sum(moles)


def moles_to_mass(model, zs: Float[Array, "n"]) -> Float[Array, "n"]:
    """Convert a mole fraction into a mass fraction."""
    mw = 1e-3 * jnp.array(model.params.Mw.values)
    mass = zs * mw
    return mass / jnp.sum(mass)


# Pressure-based bulk properties
entropy = create_jax_wrapper("entropy", "(),(),(n)->()")
internal_energy = create_jax_wrapper("internal_energy", "(),(),(n)->()")
mass_enthalpy = create_jax_wrapper("mass_enthalpy", "(),(),(n)->()")
mass_entropy = create_jax_wrapper("mass_entropy", "(),(),(n)->()")
mass_density = create_jax_wrapper("mass_density", "(),(),(n)->()")
mass_gibbs_energy = create_jax_wrapper("mass_gibbs_energy", "(),(),(n)->()")
mass_internal_energy = create_jax_wrapper("mass_internal_energy", "(),(),(n)->()")
speed_of_sound = create_jax_wrapper("speed_of_sound", "(),(),(n)->()")

# Volume-based bulk properties
pressure = create_jax_wrapper("pressure", "(),(),(n)->()")

# Multiphase properties
bubble_pressure = create_jax_wrapper("bubble_pressure", "(),(n)->(),(),(),(n)")
saturation_pressure = create_jax_wrapper("saturation_pressure", "(),()->(),(),()")
