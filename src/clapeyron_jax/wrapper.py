"""JAX-compatible wrappers for Clapeyron.jl."""

import equinox as eqx
import jax
import jax.numpy as jnp

from clapeyron_jax.utils import parse_symbolic_shape, unwrap_scalar

# from clapeyron_jax.config import JAX_ENABLE_X64


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
            args = [unwrap_scalar(arr) for arr in args]
            kwargs = {key: unwrap_scalar(arr) for key, arr in kwargs.items()}

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
