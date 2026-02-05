from __future__ import annotations

from functools import cached_property
from typing import Any

import numpy as np

from decent_bench.costs._base._cost import Cost
from decent_bench.costs._base._sum_cost import SumCost
from decent_bench.utils.array import Array
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks


class ScaledCost(Cost):
    """A scalar multiple of another cost function."""

    def __init__(self, cost: Cost, scalar: float):
        self.cost: Cost
        self.scalar: float
        if isinstance(cost, ScaledCost):
            self.cost = cost.cost
            self.scalar = scalar * cost.scalar
        else:
            self.cost = cost
            self.scalar = scalar

    @property
    def shape(self) -> tuple[int, ...]:
        return self.cost.shape

    @property
    def framework(self) -> SupportedFrameworks:
        return self.cost.framework

    @property
    def device(self) -> SupportedDevices:
        return self.cost.device

    @cached_property
    def m_smooth(self) -> float:
        if self.scalar == 0:
            return 0.0
        return float(abs(self.scalar) * self.cost.m_smooth)

    @cached_property
    def m_cvx(self) -> float:
        if self.scalar > 0:
            return float(self.scalar * self.cost.m_cvx)
        if self.scalar == 0:
            return 0.0
        return np.nan

    def function(self, x: Array, *args: Any, **kwargs: Any) -> float:  # noqa: ANN401
        return float(self.scalar * self.cost.function(x, *args, **kwargs))

    def gradient(self, x: Array, *args: Any, **kwargs: Any) -> Array:  # noqa: ANN401
        return self.cost.gradient(x, *args, **kwargs) * self.scalar

    def hessian(self, x: Array, *args: Any, **kwargs: Any) -> Array:  # noqa: ANN401
        return self.cost.hessian(x, *args, **kwargs) * self.scalar

    def proximal(self, x: Array, rho: float, *args: Any, **kwargs: Any) -> Array:  # noqa: ANN401
        if self.scalar == 0:
            return x
        return self.cost.proximal(x, rho * self.scalar, *args, **kwargs)

    def __add__(self, other: Cost) -> Cost:
        if self.shape != other.shape:
            raise ValueError(f"Mismatching domain shapes: {self.shape} vs {other.shape}")
        return SumCost([self, other])
