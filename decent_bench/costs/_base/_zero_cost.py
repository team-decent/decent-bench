from __future__ import annotations

from functools import cached_property
from typing import Any

import decent_bench.utils.interoperability as iop
from decent_bench.costs._base._cost import Cost
from decent_bench.utils.array import Array
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks


class ZeroCost(Cost):
    """A cost function that is identically zero."""

    def __init__(
        self,
        shape: tuple[int, ...],
        framework: SupportedFrameworks = SupportedFrameworks.NUMPY,
        device: SupportedDevices = SupportedDevices.CPU,
    ):
        if not all(isinstance(d, int) and d >= 0 for d in shape):
            raise ValueError("shape must be a tuple of non-negative integers")

        self._shape = shape
        self._framework = framework
        self._device = device

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def framework(self) -> SupportedFrameworks:
        return self._framework

    @property
    def device(self) -> SupportedDevices:
        return self._device

    @cached_property
    def m_smooth(self) -> float:
        return 0.0

    @cached_property
    def m_cvx(self) -> float:
        return 0.0

    def _check_shape(self, x: Array) -> None:
        if iop.shape(x) != self.shape:
            raise ValueError(f"Mismatching domain shapes: {iop.shape(x)} vs {self.shape}")

    def function(self, x: Array, **kwargs: Any) -> float:  # noqa: ANN401
        self._check_shape(x)
        return 0.0

    def gradient(self, x: Array, **kwargs: Any) -> Array:  # noqa: ANN401
        self._check_shape(x)
        return iop.zeros(self.shape, self.framework, self.device)

    def hessian(self, x: Array, **kwargs: Any) -> Array:  # noqa: ANN401
        self._check_shape(x)
        return iop.zeros(self.shape + self.shape, self.framework, self.device)

    def proximal(self, x: Array, rho: float, **kwargs: Any) -> Array:  # noqa: ANN401
        if rho <= 0:
            raise ValueError("The penalty parameter rho must be positive.")
        self._check_shape(x)
        return x

    def __add__(self, other: Cost) -> Cost:
        if not isinstance(other, Cost):
            raise TypeError(f"Cost can only be added to another Cost, got {type(other)}.")
        if self.shape != other.shape:
            raise ValueError(f"Mismatching domain shapes: {self.shape} vs {other.shape}")

        if isinstance(other, ZeroCost):
            return self

        return other
