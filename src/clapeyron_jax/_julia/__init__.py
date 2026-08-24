"""The Julia backend."""


def _load_interpreter():
    """Load and configure the Julia interpreter."""
    from juliacall import Main as jl

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

    function jvp_wrapper(func, primals, tangents, kwargs)
        # Create a tag for the Dual number
        base_type = eltype(primals[1])
        T = typeof(ForwardDiff.Tag(JAXTag(), base_type))

        args = [
            t === nothing ? p : ForwardDiff.Dual{T}.(p, t)
            for (p, t) in zip(primals, tangents)
        ]
        kwargs = Dict(Symbol(k) => v for (k, v) in kwargs)

        dual_result = func.(args...; kwargs...)

        return unwrap_dual(dual_result)
    end

    function eval_fn(func, args...; kwargs...)
        kwargs = Dict(Symbol(k) => v for (k, v) in kwargs)
        return func.(args...; kwargs...)
    end
    """)

    return jl


julia = _load_interpreter()
