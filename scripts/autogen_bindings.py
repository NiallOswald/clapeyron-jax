"""Generate JAX bindings for Clapeyron.jl."""

from juliacall import Main as jl

jl.seval("using Clapeyron")


EXCLUDED_NAMES = []


def get_objects():
    exported_symbols = jl.Base.names(jl.Clapeyron)

    funcs, models = [], []
    for sym in exported_symbols:
        sym = str(sym)

        if sym in EXCLUDED_NAMES:
            continue
        if sym.startswith("@"):
            continue

        obj = getattr(jl.Clapeyron, sym)

        if jl.isa(obj, jl.Function):
            funcs.append(obj)
        else:
            models.append(obj)

    return funcs, models
