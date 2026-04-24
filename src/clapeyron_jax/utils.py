"""Common utilities for Clapeyron.jl wrapper."""

import re

import jax
from jaxtyping import Array


def unwrap_scalar(obj):
    """Extract Python scalar from 0-d array, keep higher-d arrays as-is."""
    if isinstance(obj, Array) and obj.ndim == 0:
        return obj.item()
    return obj


def parse_symbolic_shape(signature, inputs, dtype) -> jax.ShapeDtypeStructs:
    """
    Parses a signature like '(a, b), (n) -> (n, a)'
    and returns a tuple of ShapeDtypeStruct.
    """
    in_part, out_part = signature.split("->")

    # Extract dimensions
    in_specs = re.findall(r"\((.*?)\)", in_part)
    out_specs = re.findall(r"\((.*?)\)", out_part)

    # Map symbols to integer values from input tracers
    symbol_map = {}
    for spec, arr in zip(in_specs, inputs):
        if not spec:  # input is scalar
            continue

        dims = [d.strip() for d in spec.split(",")]
        for i, dim_symbol in enumerate(dims):
            if dim_symbol.isdigit():
                continue  # skip constant dimensions
            symbol_map[dim_symbol] = arr.shape[i]

    # Build output ShapeDtypeStructs using the symbol_map
    results = []
    for spec in out_specs:
        spec = spec.strip()
        if not spec:  # output is scalar
            out_dims = ()
        else:
            out_dims = []
            for d in [d.strip() for d in spec.split(",")]:
                # Use mapped symbol or literal integer
                out_dims.append(symbol_map[d] if d in symbol_map else int(d))
            out_dims = tuple(out_dims)

        results.append(jax.ShapeDtypeStruct(out_dims, dtype))

    return tuple(results) if len(results) > 1 else results[0]
