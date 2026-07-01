"""JAX-compatible wrappers for Clapeyron.jl."""

import equinox as eqx
import jax
import jax.numpy as jnp
from juliacall import Main as jl

from clapeyron_jax.utils import parse_symbolic_shape

jl.seval("""
using Clapeyron, ForwardDiff

struct JAXTag end

unwrap_dual(x::ForwardDiff.Dual) =
    (ForwardDiff.value(x), ForwardDiff.partials(x)[1])

unwrap_dual(x::AbstractArray{<:ForwardDiff.Dual}) =
    (ForwardDiff.value.(x), getindex.(ForwardDiff.partials.(x), 1))

unwrap_dual(x::Tuple) =
    map(unwrap_dual, x) |> y -> (first.(y), last.(y))

unwrap_dual(x::NamedTuple) = begin
    y = map(unwrap_dual, values(x))
    (
        NamedTuple{keys(x)}(Tuple(first.(y))),
        NamedTuple{keys(x)}(Tuple(last.(y)))
    )
end

unwrap_dual(x) =
    throw(ArgumentError("Unsupported type: $(typeof(x))"))

function jvp_wrapper(func, model, primals, tangents, kwargs)
    # Create a tag for the Dual number
    base_type = eltype(primals[1])
    T = typeof(ForwardDiff.Tag(JAXTag(), base_type))

    args = [
        t === nothing ? p : ForwardDiff.Dual{T}.(p, t)
        for (p, t) in zip(primals, tangents)
    ]
    kwargs = Dict(Symbol(k) => v for (k, v) in kwargs)

    dual_result = func.(model, args...; kwargs...)

    return unwrap_dual(dual_result)
end
""")

jvp_wrapper = jl.jvp_wrapper


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
            _, _tangent_out = jvp_wrapper(func, model, _args, _t_args, _kwargs)
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
