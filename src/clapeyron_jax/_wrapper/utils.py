import re
from collections.abc import Sized

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
    for arg_i, (spec, arg) in enumerate(zip(in_specs, args)):
        if spec:
            keys = [d.strip() for d in spec.split(",")]
        else:
            # spec is a scalar
            keys = []

        match arg:
            case Array():
                val_dims = arg.shape
            case Sized():
                val_dims = (len(arg),)
            case int() | float():
                # arg is a scalar
                val_dims = ()
            case _:
                raise TypeError(
                    f"unexpected argument in position {arg_i}. Expected Array, Sized, "
                    f"or scalar but found {type(arg)}"
                )

        for axis_i, (key, val) in enumerate(zip(keys, val_dims, strict=True)):
            if key.isdigit():
                spec_val = int(key)
            elif key in symbol_map:
                spec_val = symbol_map[key]
            else:
                symbol_map[key] = val
                continue

            if val != spec_val:
                raise ValueError(
                    f"shape mismatch: argument in position {arg_i} has length {val} in "
                    f"axis {axis_i} but signature specifies length {spec_val}"
                )

    # Build output ShapeDtypeStructs using the symbol_map
    result_shape_dtypes = []
    for spec in out_specs:
        if spec:
            keys = [d.strip() for d in spec.split(",")]
        else:
            # spec is a scalar
            keys = []

        out_dims = tuple(
            map(
                lambda key: symbol_map[key] if key in symbol_map else int(key),
                keys,
            )
        )

        result_shape_dtypes.append(jax.ShapeDtypeStruct(out_dims, float))

    if len(result_shape_dtypes) > 1:
        return tuple(result_shape_dtypes)
    else:
        return result_shape_dtypes[0]
