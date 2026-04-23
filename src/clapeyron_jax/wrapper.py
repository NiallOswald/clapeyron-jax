"""JAX-compatible wrappers for Clapeyron.jl."""

import re

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

# from clapeyron_jax.config import JAX_ENABLE_X64


def _unwrap_scalar(obj):
    """Extract Python scalar from 0-d array, keep higher-d arrays as-is."""
    if isinstance(obj, Array) and obj.ndim == 0:
        return obj.item()
    return obj


def parse_symbolic_shape(signature, inputs, dtype):
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


def create_jax_wrapper(
    func,
    signature,
    dtype=jnp.float32,
):
    # TODO: Get default dtype from JAX config

    def wrapped(model, *args, **kwargs):
        # Find output shape using the symbolic signature
        result_shape_dtypes = parse_symbolic_shape(signature, args, dtype)

        def callback(args, kwargs):
            # Unwrap 0-d JAX arrays
            args = [_unwrap_scalar(arr) for arr in args]
            kwargs = {key: _unwrap_scalar(arr) for key, arr in kwargs.items()}

            result = func(model, *args, **kwargs)

            # Cast output result to JAX arrays
            return jax.tree_util.tree_map(
                lambda val, struct: jnp.asarray(val, dtype=struct.dtype).reshape(
                    struct.shape
                ),
                result,
                result_shape_dtypes,
            )

        return eqx.filter_pure_callback(
            callback,
            args,
            kwargs,
            result_shape_dtypes=result_shape_dtypes,
            vmap_method="sequential",
        )

    wrapped.__name__ = func.__name__
    wrapped.__doc__ = (
        f"JAX-compatible wrapper for pyclapeyron.{func.__name__}\n\n"
        f"{func.__doc__ or 'No documentation available'}"
    )

    return wrapped
