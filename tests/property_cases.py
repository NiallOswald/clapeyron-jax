"""Clapeyron.jl properties for testing."""

import clapeyron_jax as cljax


class BulkPropertyCases:
    """Bulk properties for testing."""

    def case_mass_density(self, p, T, z):
        return cljax.mass_density, (p, T, z), dict()

    def case_pressure(self, V, T, z):
        return cljax.pressure, (V, T, z), dict()
