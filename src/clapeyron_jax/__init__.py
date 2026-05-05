"""JAX-compatible wrappers for Clapeyron.jl."""

from enum import StrEnum, auto

from .wrapper import create_jax_wrapper


class Phase(StrEnum):
    """Phases accepted by Clapeyron.jl."""

    Unknown = auto()
    Liquid = auto()
    Vapor = auto()
    Solid = auto()
    Stable = auto()


# Pressure-based bulk properties
mass_density = create_jax_wrapper("mass_density", "(),(),(n)->()")

# Volume-based bulk properties
pressure = create_jax_wrapper("pressure", "(),(),(n)->()")

# Multiphase properties
bubble_pressure = create_jax_wrapper("bubble_pressure", "(),(n)->(),(),(),(n)")
saturation_pressure = create_jax_wrapper("saturation_pressure", "(),()->(),(),()")
