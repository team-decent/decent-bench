from __future__ import annotations

from functools import cached_property
from typing import Any

import numpy as np

import decent_bench.utils.interoperability as iop
from decent_bench.costs._base._cost import Cost
from decent_bench.costs._base._sum_cost import SumCost
from decent_bench.utils.array import Array
from decent_bench.utils.tags import tags
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks

__all__ = [
    "BaseRegularizerCost",
    "FractionalQuadraticRegularizerCost",
    "L1RegularizerCost",
    "L2RegularizerCost",
]


class BaseRegularizerCost(Cost):
    """Base class for element-wise regularizers defined over a vector x."""

    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        framework: SupportedFrameworks = SupportedFrameworks.NUMPY,
        device: SupportedDevices = SupportedDevices.CPU,
    ):
        if len(shape) == 0:
            raise ValueError("Regularizer shape must be non-empty.")
        if any(dim <= 0 for dim in shape):
            raise ValueError(f"Regularizer shape must be positive, got {shape}.")
        self._shape = shape
        self._dim = int(np.prod(shape))
        self._framework = framework
        self._device = device
        self._hessian_cache: Array | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def framework(self) -> SupportedFrameworks:
        return self._framework

    @property
    def device(self) -> SupportedDevices:
        return self._device

    def __add__(self, other: Cost) -> Cost:
        """
        Add another cost function, returning a sum cost.

        Raises:
            ValueError: if the domain shapes don't match.

        """
        if self.shape != other.shape:
            raise ValueError(f"Mismatching domain shapes: {self.shape} vs {other.shape}")
        return SumCost([self, other])


@tags("regularizer")
class L1RegularizerCost(BaseRegularizerCost):
    r"""
    L1 regularizer cost.

    .. math:: f(\mathbf{x}) = \|\mathbf{x}\|_1 = \sum_i |x_i|
    """

    @cached_property
    def m_smooth(self) -> float:  # pyright: ignore[reportIncompatibleMethodOverride]
        return np.nan

    @cached_property
    def m_cvx(self) -> float:  # pyright: ignore[reportIncompatibleMethodOverride]
        return 0.0

    @iop.autodecorate_cost_method(Cost.function)
    def function(self, x: Array, **kwargs: Any) -> float:  # noqa: ARG002, ANN401
        return float(iop.astype(iop.sum(iop.absolute(x)), float))

    @iop.autodecorate_cost_method(Cost.gradient)
    def gradient(self, x: Array, **kwargs: Any) -> Array:  # noqa: ARG002, ANN401
        return iop.sign(x)

    @iop.autodecorate_cost_method(Cost.hessian)
    def hessian(self, x: Array, **kwargs: Any) -> Array:  # noqa: ARG002, ANN401
        if self._hessian_cache is None:
            self._hessian_cache = iop.zeros((self._dim, self._dim), framework=self.framework, device=self.device)
        return self._hessian_cache

    @iop.autodecorate_cost_method(Cost.proximal)
    def proximal(self, x: Array, rho: float, **kwargs: Any) -> Array:  # noqa: ARG002, ANN401
        if rho <= 0:
            raise ValueError("The penalty parameter rho must be positive.")
        shrink = iop.maximum(iop.absolute(x) - rho, 0.0)
        return iop.sign(x) * shrink


@tags("regularizer")
class L2RegularizerCost(BaseRegularizerCost):
    r"""
    L2 regularizer cost.

    .. math:: f(\mathbf{x}) = \frac{1}{2}\|\mathbf{x}\|_2^2
    """

    @cached_property
    def m_smooth(self) -> float:  # pyright: ignore[reportIncompatibleMethodOverride]
        return 1.0

    @cached_property
    def m_cvx(self) -> float:  # pyright: ignore[reportIncompatibleMethodOverride]
        return 1.0

    @iop.autodecorate_cost_method(Cost.function)
    def function(self, x: Array, **kwargs: Any) -> float:  # noqa: ARG002, ANN401
        return float(iop.astype(0.5 * iop.sum(iop.mul(x, x)), float))

    @iop.autodecorate_cost_method(Cost.gradient)
    def gradient(self, x: Array, **kwargs: Any) -> Array:  # noqa: ARG002, ANN401
        return x

    @iop.autodecorate_cost_method(Cost.hessian)
    def hessian(self, x: Array, **kwargs: Any) -> Array:  # noqa: ARG002, ANN401
        if self._hessian_cache is None:
            self._hessian_cache = iop.eye(self._dim, framework=self.framework, device=self.device)
        return self._hessian_cache

    @iop.autodecorate_cost_method(Cost.proximal)
    def proximal(self, x: Array, rho: float, **kwargs: Any) -> Array:  # noqa: ARG002, ANN401
        if rho <= 0:
            raise ValueError("The penalty parameter rho must be positive.")
        return x / (1.0 + rho)


@tags("regularizer")
class FractionalQuadraticRegularizerCost(BaseRegularizerCost):
    r"""
    Nonconvex fractional quadratic regularizer.

    .. math:: f(\mathbf{x}) = \sum_i \frac{x_i^2}{1 + x_i^2}
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        framework: SupportedFrameworks = SupportedFrameworks.NUMPY,
        device: SupportedDevices = SupportedDevices.CPU,
        prox_max_iter: int = 100,
        prox_tol: float | None = 1e-10,
    ):
        super().__init__(shape, framework=framework, device=device)
        if prox_max_iter <= 0:
            raise ValueError("prox_max_iter must be positive.")
        self._prox_max_iter = prox_max_iter
        self._prox_tol = prox_tol

    @cached_property
    def m_smooth(self) -> float:  # pyright: ignore[reportIncompatibleMethodOverride]
        return 2.0

    @cached_property
    def m_cvx(self) -> float:  # pyright: ignore[reportIncompatibleMethodOverride]
        return np.nan

    @iop.autodecorate_cost_method(Cost.function)
    def function(self, x: Array, **kwargs: Any) -> float:  # noqa: ARG002, ANN401
        x2 = x * x
        return float(iop.astype(iop.sum(x2 / (1.0 + x2)), float))

    @iop.autodecorate_cost_method(Cost.gradient)
    def gradient(self, x: Array, **kwargs: Any) -> Array:  # noqa: ARG002, ANN401
        x2 = x * x
        denom = (1.0 + x2) ** 2
        return 2.0 * x / denom

    @iop.autodecorate_cost_method(Cost.hessian)
    def hessian(self, x: Array, **kwargs: Any) -> Array:  # noqa: ARG002, ANN401
        x2 = x * x
        denom = (1.0 + x2) ** 3
        second = 2.0 * (1.0 - 3.0 * x2) / denom
        return iop.diag(iop.reshape(second, (self._dim,)))

    @iop.autodecorate_cost_method(Cost.proximal)
    def proximal(self, x: Array, rho: float, **kwargs: Any) -> Array:  # noqa: ARG002, ANN401
        if rho <= 0:
            raise ValueError("The penalty parameter rho must be positive.")
        step_size = 1.0 / (2.0 + 1.0 / rho)
        current = iop.copy(x)
        for _ in range(self._prox_max_iter):
            x2 = current * current
            denom = (1.0 + x2) ** 2
            grad = 2.0 * current / denom + (current - x) / rho
            next_x = current - step_size * grad
            if self._prox_tol is not None and iop.astype(iop.norm(next_x - current), float) <= self._prox_tol:
                return next_x
            current = next_x
        return current
