"""Phases accepted by Clapeyron.jl."""

from typing import Literal, Union

type Unknown = Literal["unknown"]
type Liquid = Literal["liquid"]
type Vapor = Literal["vapor"]
type Solid = Literal["solid"]
type Stable = Literal["stable"]

type Phase = Union[Unknown, Liquid, Vapor, Solid, Stable]
