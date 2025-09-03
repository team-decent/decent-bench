from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy import float64
from numpy.typing import NDArray


class CostFunction(ABC):
    """Used by agents to evaluate the cost and its derivatives at a certain x."""

    @property
    @abstractmethod
    def domain_shape(self) -> tuple[int, ...]:
        """Required shape of x."""

    @abstractmethod
    def evaluate(self, x: NDArray[float64]) -> float:
        """Evaluate function at x."""

    @abstractmethod
    def gradient(self, x: NDArray[float64]) -> NDArray[float64]:
        """Gradient at x."""

    @abstractmethod
    def hessian(self, x: NDArray[float64]) -> NDArray[float64]:
        """Hessian at x."""

    @abstractmethod
    def proximal(self, y: NDArray[float64], rho: float) -> NDArray[float64]:
        r"""
        Proximal at y.

        The proximal operator is defined as

        .. math::
            \operatorname{prox}_{\rho f}(\mathbf{y})
            = \arg\min_{\mathbf{x}}  \left\{ f(\mathbf{x}) + \frac{1}{2\rho} \| \mathbf{x} - \mathbf{y} \|^2 \right\}

        where :math:`\rho > 0` is the penalty and :math:`f` the cost function.
        """

    @abstractmethod
    def __add__(self, other: CostFunction) -> CostFunction:
        """
        Add another cost function to create a new one.

        :class:`SumCost` be used as the result of a cost function's :meth:`__add__`. However, it's often more efficient
        to preserve the cost function type if possible. For example, the addition of two :class:`QuadraticCost`
        objects benefits from returning a new :class:`QuadraticCost` instead of a :class:`SumCost` as this
        preserves the closed form proximal solution and only requires one evaluation instead of two when calling
        :meth:`evaluate`, :meth:`gradient`, and :meth:`hessian`.
        """


class QuadraticCost(CostFunction):
    r"""
    Quadratic cost function.

    .. math:: f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^T \mathbf{Ax} + \mathbf{b}^T \mathbf{x} + c
    """

    def __init__(self, A: NDArray[float64], b: NDArray[float64], c: float):  # noqa: N803
        if A.ndim != 2:
            raise ValueError("Matrix A must be 2D")
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix A must be square")
        if b.ndim != 1:
            raise ValueError("Vector b must be 1D")
        if A.shape[0] != b.shape[0]:
            raise ValueError(f"Dimension mismatch: A has shape {A.shape} but b has length {b.shape[0]}")
        self.A = A
        self.b = b
        self.c = c

    @property
    def domain_shape(self) -> tuple[int, ...]:
        return self.b.shape

    def evaluate(self, x: NDArray[float64]) -> float:
        r"""
        Evaluate function at x.

        .. math:: \frac{1}{2} \mathbf{x}^T \mathbf{Ax} + \mathbf{b}^T \mathbf{x} + c
        """
        return float(0.5 * x.dot(self.A.dot(x)) + self.b.dot(x) + self.c)

    def gradient(self, x: NDArray[float64]) -> NDArray[float64]:
        r"""
        Gradient at x.

        .. math:: \frac{1}{2} (\mathbf{A}+\mathbf{A^T})\mathbf{x} + \mathbf{b}
        """
        return 0.5 * (self.A + self.A.T) @ x + self.b

    def hessian(self, x: NDArray[float64]) -> NDArray[float64]:  # noqa: ARG002
        r"""
        Hessian at x.

        .. math:: \frac{1}{2} (\mathbf{A}+\mathbf{A^T})
        """
        return 0.5 * (self.A + self.A.T)

    def proximal(self, y: NDArray[float64], rho: float) -> NDArray[float64]:
        r"""
        Proximal at y.

        .. math::
            (\rho \mathbf{A} + \mathbf{I})^{-1} (\mathbf{y} - \rho \mathbf{b})

        where :math:`\rho > 0` is the penalty.

        This is a closed form solution, see :meth:`CostFunction.proximal` for the general proximal definition.
        """
        lhs = self.A * rho + np.eye(self.A.shape[1])
        rhs = y - self.b * rho
        return np.asarray(np.linalg.solve(lhs, rhs), dtype=np.float64)

    def __add__(self, other: CostFunction) -> CostFunction:
        """
        Add another cost function.

        Returns:
            :class:`QuadraticCost` if *other* is :class:`QuadraticCost` or :class:`LinearRegression`,
            else :class:`SumCost`

        Raises:
            ValueError: if the domain shapes don't match

        """
        if self.domain_shape != other.domain_shape:
            raise ValueError(f"Mismatching domain shapes: {self.domain_shape} vs {other.domain_shape}")
        if isinstance(other, QuadraticCost):
            return QuadraticCost(self.A + other.A, self.b + other.b, self.c + other.c)
        if isinstance(other, LinearRegression):
            return QuadraticCost(
                self.A + other.quadratic_cf.A, self.b + other.quadratic_cf.b, self.c + other.quadratic_cf.c
            )
        return SumCost([self, other])


class LinearRegression(CostFunction):
    r"""
    Linear regression cost function.

    .. math:: f(\mathbf{x}) = \frac{1}{2} \| \mathbf{Ax} - \mathbf{b} \|^2

    or in the general quadratic form

    .. math::
        f(\mathbf{x})
        = \frac{1}{2} \mathbf{x}^T\mathbf{A}^T\mathbf{Ax}
        - (\mathbf{A}^T \mathbf{b})^T \mathbf{x}
        + \frac{1}{2} \mathbf{b}^T\mathbf{b}
    """

    def __init__(self, A: NDArray[float64], b: NDArray[float64]):  # noqa: N803
        if A.shape[0] != b.shape[0]:
            raise ValueError(f"Dimension mismatch: A has {A.shape[0]} rows but b has {b.shape[0]} elements")
        self.quadratic_cf = QuadraticCost(A.T.dot(A), -A.T.dot(b), 0.5 * b.dot(b))
        self.A = A
        self.b = b

    @property
    def domain_shape(self) -> tuple[int, ...]:
        return self.quadratic_cf.domain_shape

    def evaluate(self, x: NDArray[float64]) -> float:
        r"""
        Evaluate function at x.

        .. math:: \frac{1}{2} \| \mathbf{Ax} - \mathbf{b} \|^2
        """
        return self.quadratic_cf.evaluate(x)

    def gradient(self, x: NDArray[float64]) -> NDArray[float64]:
        r"""
        Gradient at x.

        .. math:: \mathbf{A}^T\mathbf{Ax} - \mathbf{A}^T \mathbf{b}
        """
        return self.quadratic_cf.gradient(x)

    def hessian(self, x: NDArray[float64]) -> NDArray[float64]:
        r"""
        Hessian at x.

        .. math:: \mathbf{A}^T\mathbf{A}
        """
        return self.quadratic_cf.hessian(x)

    def proximal(self, y: NDArray[float64], rho: float) -> NDArray[float64]:
        r"""
        Proximal at y.

        .. math::
            (\rho \mathbf{A}^T \mathbf{A} + \mathbf{I})^{-1} (\mathbf{y} + \rho \mathbf{A}^T\mathbf{b})

        where :math:`\rho > 0` is the penalty.

        This is a closed form solution, see :meth:`CostFunction.proximal` for the general proximal definition.
        """
        return self.quadratic_cf.proximal(y, rho)

    def __add__(self, other: CostFunction) -> CostFunction:
        """
        Add another cost function.

        Returns:
            :class:`QuadraticCost` if *other* is :class:`LinearRegression` or :class:`QuadraticCost`,
            else :class:`SumCost`

        Raises:
            ValueError: if the domain shapes don't match

        """
        if self.domain_shape != other.domain_shape:
            raise ValueError(f"Mismatching domain shapes: {self.domain_shape} vs {other.domain_shape}")
        if isinstance(other, LinearRegression):
            return QuadraticCost(
                self.quadratic_cf.A + other.quadratic_cf.A,
                self.quadratic_cf.b + other.quadratic_cf.b,
                self.quadratic_cf.c + other.quadratic_cf.c,
            )
        if isinstance(other, QuadraticCost):
            return QuadraticCost(
                self.quadratic_cf.A + other.A, self.quadratic_cf.b + other.b, self.quadratic_cf.c + other.c
            )
        return SumCost([self, other])


class SumCost(CostFunction):
    """The sum of multiple cost functions."""

    def __init__(self, cost_functions: list[CostFunction]):
        if not all(cost_functions[0].domain_shape == cf.domain_shape for cf in cost_functions):
            raise ValueError("All cost functions must have the same domain shape")
        self.cost_functions = cost_functions

    @property
    def domain_shape(self) -> tuple[int, ...]:
        return self.cost_functions[0].domain_shape

    def evaluate(self, x: NDArray[float64]) -> float:
        """Sum the :meth:`evaluate` of each cost function."""
        return sum(cf.evaluate(x) for cf in self.cost_functions)

    def gradient(self, x: NDArray[float64]) -> NDArray[float64]:
        """Sum the :meth:`gradient` of each cost function."""
        return np.asarray(np.sum([cf.gradient(x) for cf in self.cost_functions], axis=0))

    def hessian(self, x: NDArray[float64]) -> NDArray[float64]:
        """Sum the :meth:`hessian` of each cost function."""
        return np.asarray(np.sum([cf.hessian(x) for cf in self.cost_functions], axis=0))

    def proximal(self, y: NDArray[float64], rho: float) -> NDArray[float64]:
        """
        Proximal at y solved using an iterative method.

        See :meth:`CostFunction.proximal` for the general proximal definition.
        """
        raise NotImplementedError

    def __add__(self, other: CostFunction) -> SumCost:
        """Add another cost function."""
        return SumCost([*self.cost_functions, other])


class ProximalCost(CostFunction):
    r"""
    The function minimized by the proximal operator.

    .. math::
            f_{prox}(\mathbf{x}) = f(\mathbf{x}) + \frac{1}{2\rho} \| \mathbf{x} - \mathbf{y} \|^2

    where :math:`f`, :math:`\mathbf{y}`, and :math:`\rho` are the cost function, input, and penalty respectively,
    all fixed by the proximal operator.

    See :meth:`CostFunction.proximal` for how the proximal operator is defined and relates to this function.
    """

    def __init__(self, f: CostFunction, y: NDArray[float64], rho: float):
        if len(f.domain_shape) > 1 or len(y.shape) > 1:
            raise ValueError("Shape of cost function domain and y must have exactly one axis")
        if f.domain_shape != y.shape:
            raise ValueError("Cost function domain and y need to have the same shape")
        if rho <= 0:
            raise ValueError("Penalty term `rho` must be greater than 0")
        self.inner = f + QuadraticCost(A=np.eye(len(y)) / rho, b=-y / rho, c=y.dot(y) / (2 * rho))

    @property
    def domain_shape(self) -> tuple[int, ...]:
        return self.inner.domain_shape

    def evaluate(self, x: NDArray[float64]) -> float:
        r"""
        Evaluate function at x.

        .. math:: f(\mathbf{x}) + \frac{1}{2\rho} \| \mathbf{x} - \mathbf{y} \|^2
        """
        return self.inner.evaluate(x)

    def gradient(self, x: NDArray[float64]) -> NDArray[float64]:
        r"""
        Gradient at x.

        .. math:: \nabla f(\mathbf{x}) + \frac{1}{\rho} (\mathbf{x} - \mathbf{y})
        """
        return self.inner.gradient(x)

    def hessian(self, x: NDArray[float64]) -> NDArray[float64]:
        r"""
        Hessian at x.

        .. math:: \nabla^2 f(\mathbf{x}) + \frac{1}{\rho} \mathbf{I}
        """
        return self.inner.hessian(x)

    def proximal(self, y: NDArray[float64], rho: float) -> NDArray[float64]:
        """
        Proximal at y solved using an iterative method.

        See :meth:`CostFunction.proximal` for the general proximal definition.
        """
        raise NotImplementedError

    def __add__(self, other: CostFunction) -> CostFunction:
        """Add another cost function."""
        return SumCost([self, other])
