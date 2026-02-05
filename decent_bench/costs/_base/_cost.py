from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Real
from typing import Any

from decent_bench.utils.array import Array
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks


class Cost(ABC):
    """Used by agents to evaluate the cost and its derivatives at a certain x."""

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Required shape of x."""

    @property
    def domain_shape(self) -> tuple[int, ...]:
        """Alias for :attr:`shape`."""
        return self.shape

    @property
    @abstractmethod
    def framework(self) -> SupportedFrameworks:
        """
        The framework used by this cost function.

        Make sure that all :class:`decent_bench.utils.array.Array` objects returned by this cost function's methods
        use this framework.

        """

    @property
    @abstractmethod
    def device(self) -> SupportedDevices:
        """
        The device used by this cost function.

        Make sure that all :class:`decent_bench.utils.array.Array` objects returned by this cost function's methods
        use this device.

        """

    @property
    @abstractmethod
    def m_smooth(self) -> float:
        r"""
        Lipschitz constant of the cost function's gradient.

        The gradient's Lipschitz constant m_smooth is the smallest value such that

        .. math::
            \| \nabla f(\mathbf{x_1}) - \nabla f(\mathbf{x_2}) \| \leq m_{\text{smooth}}
            \cdot \|\mathbf{x_1} - \mathbf{x_2}\|

        for all :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`.

        Returns:
            - non-negative finite number if function is L-smooth
            - ``np.inf`` if function is differentiable everywhere but not L-smooth
            - ``np.nan`` if function is not differentiable everywhere

        """

    @property
    @abstractmethod
    def m_cvx(self) -> float:
        r"""
        Convexity constant of the cost function.

        The convexity constant m_cvx is the largest value such that

        .. math::
            f(\mathbf{x_1}) \geq f(\mathbf{x_2})
            + \nabla f(\mathbf{x_2})^T (\mathbf{x_1} - \mathbf{x_2})
            + \frac{m_{\text{cvx}}}{2} \|\mathbf{x_1} - \mathbf{x_2}\|^2

        for all :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`.

        Returns:
            - positive finite number if function is strongly convex
            - ``0`` if function is convex but not strongly convex
            - ``np.nan`` if function is not guaranteed to be convex

        """

    @abstractmethod
    def function(self, x: Array, **kwargs: Any) -> float:  # noqa: ANN401
        """Evaluate function at x."""

    def evaluate(self, x: Array, **kwargs: Any) -> float:  # noqa: ANN401
        """Alias for :meth:`function`."""
        return self.function(x, **kwargs)

    def loss(self, x: Array, **kwargs: Any) -> float:  # noqa: ANN401
        """Alias for :meth:`function`."""
        return self.function(x, **kwargs)

    def f(self, x: Array, **kwargs: Any) -> float:  # noqa: ANN401
        """Alias for :meth:`function`."""
        return self.function(x, **kwargs)

    @abstractmethod
    def gradient(self, x: Array, **kwargs: Any) -> Array:  # noqa: ANN401
        """Gradient at x."""

    @abstractmethod
    def hessian(self, x: Array, **kwargs: Any) -> Array:  # noqa: ANN401
        """Hessian at x."""

    @abstractmethod
    def proximal(self, x: Array, rho: float, **kwargs: Any) -> Array:  # noqa: ANN401
        r"""
        Proximal at x.

        The proximal operator is defined as:

        .. include:: snippets/proximal_operator.rst

        If the cost function's proximal does not have a closed form solution, it can be solved iteratively using
        :meth:`~decent_bench.centralized_algorithms.proximal_solver`.
        """

    @abstractmethod
    def __add__(self, other: Cost) -> Cost:
        """
        Add another cost function to create a new one.

        :class:`~decent_bench.costs.SumCost` can be used as the result of :meth:`__add__` by returning
        ``SumCost([self, other])``. However, it's often more efficient to preserve the cost function type if possible.
        For example, the addition of two :class:`~decent_bench.costs.QuadraticCost` objects benefits
        from returning a new :class:`~decent_bench.costs.QuadraticCost` instead of a
        :class:`~decent_bench.costs.SumCost` as this preserves the closed
        form proximal solution and only requires one evaluation instead of two when calling :meth:`evaluate`,
        :meth:`gradient`, and :meth:`hessian`.
        """

    def __mul__(self, other: Real) -> Cost:
        """Multiply by a scalar to create a weighted cost."""
        if not self._is_valid_scalar(other):
            raise TypeError(f"Cost can only be multiplied by a real number, got {type(other)}.")
        from decent_bench.costs._base._scaled_cost import ScaledCost

        return ScaledCost(self, float(other))

    def __rmul__(self, other: Real) -> Cost:
        """Right-side scalar multiplication."""
        return self * other

    def __truediv__(self, other: Real) -> Cost:
        """Divide by a scalar."""
        if not self._is_valid_scalar(other):
            raise TypeError(f"Cost can only be divided by a real number, got {type(other)}.")
        if other == 0:
            raise ZeroDivisionError("Division by zero is not allowed for Cost objects.")
        return self * (1.0 / float(other))

    def __rtruediv__(self, other: Real) -> Cost:
        """Right-side scalar division, equivalent to dividing this cost by the scalar."""
        return self / other

    def __neg__(self) -> Cost:
        """Negate this cost function."""
        return -1.0 * self

    def __sub__(self, other: Cost) -> Cost:
        """Subtract another cost function as sum with its negation."""
        if not isinstance(other, Cost):
            raise TypeError(f"Cost can only be subtracted by another Cost, got {type(other)}.")
        return self + (-other)

    def __radd__(self, other: object) -> Cost:
        """Right-side addition, used to make sum(costs) work."""
        if other == 0:
            return self
        if isinstance(other, Cost):
            return other + self
        raise TypeError(f"Cost can only be added to another Cost, got {type(other)}.")

    @staticmethod
    def _is_valid_scalar(value: Any) -> bool:
        """Return True if value is a real scalar and not bool."""
        return isinstance(value, Real) and not isinstance(value, bool)
