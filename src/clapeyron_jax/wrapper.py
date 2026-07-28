"""JAX-compatible wrappers for Clapeyron.jl."""

from functools import cached_property

import equinox as eqx
import jax
import jax.numpy as jnp

from clapeyron_jax.utils import parse_symbolic_shape

from ._julia import julia as jl


class create_jax_wrapper:
    """Wraps a Clapeyron.jl function."""

    def __init__(self, fn_name: str, signature: str):
        self.fn_name = fn_name
        self.signature = signature

    @property
    def _jl_fn(self):
        return getattr(jl.Clapeyron, self.fn_name)

    def _callback(self, args, kwargs):
        # TODO: Assumes that args[-1] == zs
        *args, zs = args

        match zs.shape:
            case (_,):
                # Not broadcasting: zs (1d) -> [zs]
                result = jl.eval_fn(self._jl_fn, *args, (zs,), **kwargs)[0]
                return jax.tree.map(lambda xs: jnp.array(xs), result)
            case (1, _):
                # Not broadcasting on zs: zs (2d) -> [*zs] == [zs[0]]
                result = jl.eval_fn(self._jl_fn, *args, (zs[0],), **kwargs)
                return jax.tree.map(lambda xs: jnp.array(xs), result)
            case (_, _):
                # Broadcasting on zs: zs (2d) -> [*zs]
                result = jl.eval_fn(self._jl_fn, *args, [*zs], **kwargs)
                return jax.tree.map(lambda *xs: jnp.array(xs), *result)
            case _:
                raise ValueError("Shape mismatch in wrapped callback")

    def _jvp_callback(self, primals, tangents, kwargs):
        _, tangent_out = jl.jvp_wrapper(self._jl_fn, primals, tangents, kwargs)
        return jax.tree_util.tree_map(lambda val: jnp.asarray(val), tangent_out)

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
                vmap_method="expand_dims",
            )

        @wrapped.def_jvp
        def wrapped_jvp(primals, tangents, **kwargs):
            # Find output shape using the symbolic signature
            result_shape_dtypes = parse_symbolic_shape(self.signature, tangents[1:])

            primal_out = wrapped(*primals, **kwargs)
            tangent_out = eqx.filter_pure_callback(
                self._jvp_callback,
                primals,
                tangents,
                kwargs,
                result_shape_dtypes=result_shape_dtypes,
                vmap_method="expand_dims",
            )

            return primal_out, tangent_out

        return wrapped

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)
