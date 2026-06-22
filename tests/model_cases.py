"""Clapeyron.jl models for testing."""

# TODO: Deprecate pyclapeyron
import pyclapeyron as cl


class SpeciesCases:
    """Species for testing."""

    def case_pure(self) -> list[str]:
        return ["water"]

    def case_binary(self) -> list[str]:
        return ["water", "ethanol"]


class ModelCases:
    """Clapeyron.jl models for testing."""

    def case_peng_robinson(self):
        return cl.PR

    def case_pcsaft(self):
        return cl.PCSAFT
