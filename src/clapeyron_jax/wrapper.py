"""JAX-compatible wrappers for Clapeyron.jl."""

import equinox as eqx
import jax
import jax.numpy as jnp
from juliacall import Main as jl

from clapeyron_jax.utils import parse_symbolic_shape, unwrap_scalar

jl.seval("using Clapeyron, ForwardDiff")

jl.seval("""
struct JAXTag end

function jvp_wrapper(func, model, primals, tangents, kwargs)
    # Create a tag for the Dual number
    base_type = eltype(primals[1])
    T = typeof(ForwardDiff.Tag(JAXTag(), base_type))

    args = [
        t !== nothing ? ForwardDiff.Dual{T}.(p, t) : p
        for (p, t) in zip(primals, tangents)
    ]

    kwargs = Dict(Symbol(k) => v for (k, v) in kwargs)

    dual_result = func(model, args...; kwargs...)

    out = []
    for item in dual_result
        if item isa AbstractArray
            push!(out, ForwardDiff.partials.(item, 1))
        else
            push!(out, ForwardDiff.partials(item, 1))
        end
    end

    return length(out) == 1 ? out[1] : Tuple(out)
end
""")

jvp_wrapper = jl.jvp_wrapper


def create_jax_wrapper(
    func_name: str,
    signature: str,
    dtype=jnp.float32,
):
    # TODO: Get default dtype from JAX config

    func = getattr(jl.Clapeyron, func_name)

    @eqx.filter_custom_jvp
    def wrapped(model, *args, **kwargs):
        # Find output shape using the symbolic signature
        result_shape_dtypes = parse_symbolic_shape(signature, args, dtype)

        def callback(args, kwargs):
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

    @wrapped.def_jvp
    def wrapped_jvp(primals, tangents, **kwargs):
        model, *args = primals
        _, *t_args = tangents

        # Compute primal output
        primal_out = wrapped(*primals, **kwargs)

        # Find output shape using the symbolic signature
        result_shape_dtypes = parse_symbolic_shape(signature, args, dtype)

        def callback(args, t_args, kwargs):
            args = [unwrap_scalar(arr) for arr in args]
            t_args = [unwrap_scalar(arr) for arr in t_args]
            kwargs = {key: unwrap_scalar(arr) for key, arr in kwargs.items()}

            tangent_out = jvp_wrapper(func, model, args, t_args, kwargs)

            return jax.tree_util.tree_map(
                lambda val, struct: jnp.asarray(val, dtype=struct.dtype).reshape(
                    struct.shape
                ),
                tangent_out,
                result_shape_dtypes,
            )

        tangent_out = eqx.filter_pure_callback(
            callback,
            args,
            t_args,
            kwargs,
            result_shape_dtypes=result_shape_dtypes,
            vmap_method="sequential",
        )

        return primal_out, tangent_out

    wrapped.__name__ = func.__name__
    wrapped.__doc__ = (
        f"JAX-compatible wrapper for pyclapeyron.{func.__name__}\n\n"
        f"{func.__doc__ or 'No documentation available'}"
    )

    return wrapped
