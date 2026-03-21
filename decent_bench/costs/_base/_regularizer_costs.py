from __future__ import annotations

from functools import cached_property
from typing import Any

import numpy as np

import decent_bench.utils.interoperability as iop
from decent_bench.costs._base._cost import Cost
from decent_bench.costs._base._sum_cost import SumCost
from decent_bench.utils.array import Array
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks

__all__ = [
    "BaseRegularizerCost",
    "FractionalQuadraticRegularizerCost",
    "L1RegularizerCost",
    "L2RegularizerCost",
]


class BaseRegularizerCost(Cost):
    """
    Base class for regularizers with regularizer-preserving arithmetic.

    Adding, subtracting, negating, scaling, or dividing regularizers returns a regularizer-aware composite instead of
    generic :class:`~decent_bench.costs.SumCost` or :class:`~decent_bench.costs.ScaledCost`. Mixing a regularizer
    with an arbitrary non-regularizer still falls back to generic cost composition.
    """

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
        if isinstance(other, BaseRegularizerCost):
            return _CompositeRegularizerCost([self, other])
        return SumCost([self, other])

    def __mul__(self, other: float) -> Cost:
        """Multiply by a scalar while preserving the regularizer abstraction."""
        if not self._is_valid_scalar(other):
            raise TypeError(f"Cost can only be multiplied by a real number, got {type(other)}.")
        return _CompositeRegularizerCost([self], weights=[float(other)])

    def __truediv__(self, other: float) -> Cost:
        """Divide by a scalar while preserving the regularizer abstraction."""
        if not self._is_valid_scalar(other):
            raise TypeError(f"Cost can only be divided by a real number, got {type(other)}.")
        if other == 0:
            raise ZeroDivisionError("Division by zero is not allowed for Cost objects.")
        return self.__mul__(1.0 / float(other))

    def __neg__(self) -> Cost:
        """Negate this regularizer while preserving the regularizer abstraction."""
        return self.__mul__(-1.0)

    def __sub__(self, other: Cost) -> Cost:
        """Subtract another cost, preserving the regularizer abstraction when possible."""
        if not isinstance(other, Cost):
            raise TypeError(f"Cost can only be subtracted by another Cost, got {type(other)}.")
        if isinstance(other, BaseRegularizerCost):
            if self.shape != other.shape:
                raise ValueError(f"Mismatching domain shapes: {self.shape} vs {other.shape}")
            return _CompositeRegularizerCost([self, other], weights=[1.0, -1.0])
        return super().__sub__(other)


class _CompositeRegularizerCost(BaseRegularizerCost):
    """
    Weighted combination of regularizers that preserves the regularizer abstraction.

    This wrapper represents sums and scalar rescalings of regularizers while keeping the
    :class:`BaseRegularizerCost` interface. It combines function, gradient, and Hessian termwise. A generic proximal
    is intentionally not implemented except for the single positively scaled regularizer case.

    Instances keep references to the wrapped cost objects. No implicit copying is performed; use
    :func:`copy.deepcopy` explicitly if independent objects are required.
    """

    def __init__(self, regularizers: list[BaseRegularizerCost], weights: list[float] | None = None):
        if len(regularizers) == 0:
            raise ValueError("Composite regularizer must contain at least one regularizer.")
        first = regularizers[0]
        super().__init__(first.shape, framework=first.framework, device=first.device)

        if weights is None:
            weights = [1.0] * len(regularizers)
        if len(regularizers) != len(weights):
            raise ValueError("Composite regularizer weights must match the number of regularizers.")

        self._terms: list[tuple[BaseRegularizerCost, float]] = []
        for regularizer, weight in zip(regularizers, weights, strict=True):
            if not isinstance(regularizer, BaseRegularizerCost):
                raise TypeError(f"Composite regularizer can only contain regularizers, got {type(regularizer)}.")
            if regularizer.shape != self.shape:
                raise ValueError(f"Mismatching domain shapes: {regularizer.shape} vs {self.shape}")
            if regularizer.framework != self.framework:
                raise ValueError(f"Mismatching frameworks: {regularizer.framework} vs {self.framework}")
            if regularizer.device != self.device:
                raise ValueError(f"Mismatching devices: {regularizer.device} vs {self.device}")
            if isinstance(regularizer, _CompositeRegularizerCost):
                for inner_regularizer, inner_weight in regularizer._terms:
                    self._terms.append((inner_regularizer, float(weight) * inner_weight))
            else:
                self._terms.append((regularizer, float(weight)))

    @cached_property
    def m_smooth(self) -> float:  # pyright: ignore[reportIncompatibleMethodOverride]
        m_smooth_vals = [abs(weight) * regularizer.m_smooth for regularizer, weight in self._terms]
        return np.nan if any(np.isnan(v) for v in m_smooth_vals) else float(sum(m_smooth_vals))

    @cached_property
    def m_cvx(self) -> float:  # pyright: ignore[reportIncompatibleMethodOverride]
        if any(weight < 0 for _, weight in self._terms):
            return np.nan
        m_cvx_vals = [weight * regularizer.m_cvx for regularizer, weight in self._terms]
        return np.nan if any(np.isnan(v) for v in m_cvx_vals) else float(sum(m_cvx_vals))

    @iop.autodecorate_cost_method(Cost.function)
    def function(self, x: Array, **kwargs: Any) -> float:  # noqa: ANN401
        return float(sum(weight * regularizer.function(x, **kwargs) for regularizer, weight in self._terms))

    @iop.autodecorate_cost_method(Cost.gradient)
    def gradient(self, x: Array, **kwargs: Any) -> Array:  # noqa: ANN401
        return iop.sum(
            iop.stack([regularizer.gradient(x, **kwargs) * weight for regularizer, weight in self._terms]),
            dim=0,
        )

    @iop.autodecorate_cost_method(Cost.hessian)
    def hessian(self, x: Array, **kwargs: Any) -> Array:  # noqa: ANN401
        return iop.sum(
            iop.stack([regularizer.hessian(x, **kwargs) * weight for regularizer, weight in self._terms]),
            dim=0,
        )

    @iop.autodecorate_cost_method(Cost.proximal)
    def proximal(self, x: Array, rho: float, **kwargs: Any) -> Array:  # noqa: ANN401
        """
        Proximal is only supported for a single positively scaled regularizer term.

        For sums of regularizers or negative scaling, composing or summing individual proximal operators is not
        mathematically valid in general.
        """
        if rho <= 0:
            raise ValueError("The penalty parameter rho must be positive.")
        if len(self._terms) == 1:
            regularizer, weight = self._terms[0]
            if weight > 0:
                return regularizer.proximal(x, rho * weight, **kwargs)
        raise NotImplementedError(
            "Composite regularizers do not implement a generic proximal operator because sums of regularizers and "
            "negative scaling do not admit a proximal from simple composition in general. Use a specialized proximal "
            "if available."
        )


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
