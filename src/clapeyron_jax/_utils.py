"""Common utilities for Clapeyron.jl wrapper."""

import re

import jax
from jaxtyping import Array


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


def parse_symbolic_shape(signature: str, args: tuple):
    """
    Parses a signature like '(a, b), (n) -> (n, a)'
    and returns a tuple of ShapeDtypeStruct.
    """
    # Extract dimensions
    in_part, out_part = signature.split("->")
    in_specs = re.findall(r"\((.*?)\)", in_part)
    out_specs = re.findall(r"\((.*?)\)", out_part)

    # Map symbols to integer values from input tracers
    batch_dims = ()
    symbol_map = {}
    for spec, arg in zip(in_specs, args):
        if not spec:  # spec is scalar
            if isinstance(arg, Array):
                batch_dims = arg.shape
            continue

        keys = [d.strip() for d in spec.split(",")]

        batch_dims = arg.shape[: -len(keys)]
        val_dims = arg.shape[-len(keys) :]

        for key, val in zip(keys, val_dims):
            if key.isdigit() or key in symbol_map:
                continue
            symbol_map[key] = val

    # Build output ShapeDtypeStructs using the symbol_map
    result_shape_dtypes = []
    for spec in out_specs:
        if not spec:
            out_dims = ()

        else:
            keys = [d.strip() for d in spec.split(",")]
            out_dims = tuple(
                map(
                    # Use mapped symbol or literal integer
                    lambda key: symbol_map[key] if key in symbol_map else int(key),
                    keys,
                )
            )

        result_shape_dtypes.append(jax.ShapeDtypeStruct(batch_dims + out_dims, float))

    if len(result_shape_dtypes) > 1:
        return tuple(result_shape_dtypes)
    else:
        return result_shape_dtypes[0]
