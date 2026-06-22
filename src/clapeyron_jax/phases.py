"""Phases accepted by Clapeyron.jl."""


class Phase(str):
    def __new__(cls):
        return super().__new__(cls, cls.__name__)


class Unknown(Phase): ...


class Liquid(Phase): ...


class Vapor(Phase): ...


class Solid(Phase): ...


class Stable(Phase): ...
