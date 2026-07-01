"""JAX-compatible wrappers for Clapeyron.jl."""

import equinox as eqx
import jax
import jax.numpy as jnp
from ._julia import julia as jl

from clapeyron_jax.utils import parse_symbolic_shape


def create_jax_wrapper(func_name: str, signature: str):
    """Wraps a Clapeyron.jl function."""
    func = getattr(jl.Clapeyron, func_name)

    @eqx.filter_custom_jvp
    def wrapped(model, *args, **kwargs):
        # Find output shape using the symbolic signature
        result_shape_dtypes = parse_symbolic_shape(signature, args)

        def callback(_args, _kwargs):
            _result = jl.broadcast(func, model, *_args, **_kwargs)
            return jax.tree_util.tree_map(lambda val: jnp.asarray(val), _result)

        return eqx.filter_pure_callback(
            callback,
            args,
            kwargs,
            result_shape_dtypes=result_shape_dtypes,
            vmap_method="expand_dims",
        )

    @wrapped.def_jvp
    def wrapped_jvp(primals, tangents, **kwargs):
        model, *args = primals
        _, *t_args = tangents

        # Compute primal output
        primal_out = wrapped(*primals, **kwargs)

        # Find output shape using the symbolic signature
        result_shape_dtypes = parse_symbolic_shape(signature, t_args)

        def callback(_args, _t_args, _kwargs):
            _, _tangent_out = jl.jvp_wrapper(func, model, _args, _t_args, _kwargs)
            return jax.tree_util.tree_map(lambda val: jnp.asarray(val), _tangent_out)

        tangent_out = eqx.filter_pure_callback(
            callback,
            args,
            t_args,
            kwargs,
            result_shape_dtypes=result_shape_dtypes,
            vmap_method="expand_dims",
        )

        return primal_out, tangent_out

    wrapped.__name__ = func.__name__
    wrapped.__doc__ = (
        f"JAX-compatible wrapper for pyclapeyron.{func.__name__}\n\n"
        f"{func.__doc__ or 'No documentation available'}"
    )

    return wrapped
