from functools import cached_property
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from clapeyron_jax._julia import julia as jl

from .utils import parse_symbolic_shape, unwrap_scalar


class JAXWrapper(eqx.Module):
    """Wraps a Clapeyron.jl function."""

    jl_fn: Callable
    signature: str = eqx.field(static=True)

    def _callback(self, args, kwargs):
        args = jax.tree.map(unwrap_scalar, args)
        result = self.jl_fn(*args, **kwargs)
        return jax.tree.map(lambda xs: jnp.array(xs), result)

    def _jvp_callback(self, primals, tangents, kwargs):
        primals = jax.tree.map(unwrap_scalar, primals)
        tangents = jax.tree.map(unwrap_scalar, tangents)
        _, tangent_out = jl.jvp_wrapper(self.jl_fn, primals, tangents, kwargs)
        return jax.tree.map(lambda val: jnp.asarray(val), tangent_out)

    @cached_property
    def fn(self):
        @eqx.filter_custom_jvp
        def wrapped(*args, **kwargs):
            # Find output shape using the symbolic signature
            result_shape_dtypes = parse_symbolic_shape(self.signature, args[1:])

            return eqx.filter_pure_callback(
                self._callback,
                args,
                kwargs,
                result_shape_dtypes=result_shape_dtypes,
                vmap_method="sequential",
            )

        @wrapped.def_jvp
        def wrapped_jvp(primals, tangents, **kwargs):
            # Find output shape using the symbolic signature
            result_shape_dtypes = parse_symbolic_shape(
                self.signature, tangents[1:], primals[1:]
            )

            primal_out = wrapped(*primals, **kwargs)
            tangent_out = eqx.filter_pure_callback(
                self._jvp_callback,
                primals,
                tangents,
                kwargs,
                result_shape_dtypes=result_shape_dtypes,
                vmap_method="sequential",
            )

            return primal_out, tangent_out

        return wrapped

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    @property
    def __wrapped__(self):
        return self.jl_fn


def create_jax_wrapper(fn: Callable, signature: str):
    return eqx.module_update_wrapper(JAXWrapper(fn, signature))
