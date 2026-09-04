import re

import jax
from jaxtyping import Array


def unwrap_scalar(obj):
    """Extract Python scalar from 0-d array, keep higher-d arrays as-is."""
    if isinstance(obj, Array) and obj.ndim == 0:
        return obj.item()
    return obj


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
    symbol_map = {}
    for spec, arg in zip(in_specs, args):
        if not spec:
            # spec is a scalar
            continue

        keys = [d.strip() for d in spec.split(",")]

        if isinstance(arg, Array):
            val_dims = arg.shape
        else:
            val_dims = (len(arg),)

        for key, val in zip(keys, val_dims):
            if key.isdigit():
                assert key == val
            elif key in symbol_map:
                assert symbol_map[key] == val
            else:
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

        result_shape_dtypes.append(jax.ShapeDtypeStruct(out_dims, float))

    if len(result_shape_dtypes) > 1:
        return tuple(result_shape_dtypes)
    else:
        return result_shape_dtypes[0]
