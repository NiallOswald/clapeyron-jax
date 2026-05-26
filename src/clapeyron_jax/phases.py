"""Phases accepted by Clapeyron.jl."""


class BasePhase(str):
    def __new__(cls):
        return super().__new__(cls, cls.__name__)


class Unknown(BasePhase):
    pass


class Liquid(BasePhase):
    pass


class Vapor(BasePhase):
    pass


class Solid(BasePhase):
    pass


class Stable(BasePhase):
    pass


Phase = Unknown | Liquid | Vapor | Solid | Stable
