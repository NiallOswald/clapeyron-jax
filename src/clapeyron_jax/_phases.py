"""Phases accepted by Clapeyron.jl."""

from ._utils import SingletonMeta


class Phase(str, metaclass=SingletonMeta):
    """A phase accepted by Clapeyron.jl."""

    def __new__(cls):
        return super().__new__(cls, cls.__name__.lower())

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return (self.__class__, ())


class Unknown(Phase): ...


class Liquid(Phase): ...


class Vapor(Phase): ...


class Solid(Phase): ...


class Stable(Phase): ...
