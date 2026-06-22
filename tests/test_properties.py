import equinox as eqx
from property_cases import BulkPropertyCases
from pytest_cases import parametrize_with_cases


@parametrize_with_cases("prop_case", cases=BulkPropertyCases)
def test_python(model, prop_case):
    prop_func, args, kwargs = prop_case
    res = prop_func(model, *args, **kwargs)


@parametrize_with_cases("prop_case", cases=BulkPropertyCases)
def test_jax(model, prop_case):
    prop_func, args, kwargs = prop_case
    jax_func = eqx.filter_jit(prop_func)
    res = jax_func(model, *args, **kwargs)
