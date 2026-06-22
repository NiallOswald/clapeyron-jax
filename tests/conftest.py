import numpy as np
from model_cases import ModelCases, SpeciesCases
from pytest_cases import fixture, parametrize_with_cases


@fixture(scope="package")
@parametrize_with_cases("model_type", cases=ModelCases)
@parametrize_with_cases("species", cases=SpeciesCases)
def model(model_type, species: list[str]):
    return model_type(species)


@fixture
def p() -> float:
    """Test pressure."""
    return 1e5  # 100 kPA


@fixture
def V() -> float:
    """Test volume."""
    return 0.025  # 0.025 m^3


@fixture
def T() -> float:
    """Test temperature."""
    return 303.15  # 30 C


@fixture
def z(model):
    """Test molar fractions."""
    n_species = len(model.components)
    return np.array([1 / n_species] * n_species)
