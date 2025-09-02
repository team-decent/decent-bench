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
    def proximal(self, x: NDArray[float64], penalty: float) -> NDArray[float64]:
        r"""
        Proximal at x.

        The proximal operator is defined as

        .. math::
            \operatorname{prox}_{\rho f}(\pmb{x})
            = \arg\min_{\pmb{y}}  \left\{ f(\pmb{y}) + \frac{1}{2\rho} \| \pmb{y} - \pmb{x} \|^2 \right\}

        where :math:`\rho > 0` is the penalty and :math:`f` the cost function.
        """


class QuadraticCostFunction(CostFunction):
    r"""
    Quadratic cost function.

    .. math:: f(\pmb{x}) = \frac{1}{2} \pmb{x}^T \pmb{Ax} + \pmb{b}^T \pmb{x} + c
    """

    def __init__(self, A: NDArray[float64], b: NDArray[float64], c: float) -> None:  # noqa: N803
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
    def domain_shape(self) -> tuple[int, ...]:  # noqa: D102
        return self.b.shape

    def evaluate(self, x: NDArray[float64]) -> float:
        r"""
        Evaluate function at x.

        .. math:: \frac{1}{2} \pmb{x}^T \pmb{Ax} + \pmb{b}^T \pmb{x} + c
        """
        return float(0.5 * x.dot(self.A.dot(x)) + self.b.dot(x) + self.c)

    def gradient(self, x: NDArray[float64]) -> NDArray[float64]:
        r"""
        Gradient at x.

        .. math:: \pmb{Ax} + \pmb{b}
        """
        return self.A @ x + self.b

    def hessian(self, x: NDArray[float64]) -> NDArray[float64]:  # noqa: ARG002
        r"""
        Hessian at x.

        .. math:: \pmb{A}
        """
        return self.A

    def proximal(self, x: NDArray[float64], penalty: float) -> NDArray[float64]:
        r"""
        Proximal at x.

        .. math::
            (\rho \pmb{A} + \pmb{I})^{-1} (\pmb{x} - \rho \pmb{b})

        This is a closed form solution, see :meth:`CostFunction.proximal` for the general proximal definition.
        """
        lhs = self.A * penalty + np.eye(self.A.shape[1])
        rhs = x - self.b * penalty
        return np.asarray(np.linalg.solve(lhs, rhs), dtype=np.float64)


class LinearRegression(CostFunction):
    r"""
    Linear regression cost function.

    .. math:: f(\pmb{x}) = \frac{1}{2} \| \pmb{Ax} - \pmb{b} \|^2

    or in the general quadratic form

    .. math::
        f(\pmb{x})
        = \frac{1}{2} \pmb{x}^T\pmb{A}^T\pmb{Ax} - (\pmb{A}^T \pmb{b})^T \pmb{x} + \frac{1}{2} \pmb{b}^T\pmb{b}
    """

    def __init__(self, A: NDArray[float64], b: NDArray[float64]) -> None:  # noqa: N803
        if A.shape[0] != b.shape[0]:
            raise ValueError(f"Dimension mismatch: A has {A.shape[0]} rows but b has {b.shape[0]} elements")
        self._quadratic_cf = QuadraticCostFunction(A.T.dot(A), -A.T.dot(b), 0.5 * b.dot(b))
        self.A = A
        self.b = b

    @property
    def domain_shape(self) -> tuple[int, ...]:  # noqa: D102
        return self._quadratic_cf.domain_shape

    def evaluate(self, x: NDArray[float64]) -> float:
        r"""
        Evaluate function at x.

        .. math:: \frac{1}{2} \| \pmb{Ax} - \pmb{b} \|^2
        """
        return self._quadratic_cf.evaluate(x)

    def gradient(self, x: NDArray[float64]) -> NDArray[float64]:
        r"""
        Gradient at x.

        .. math:: \pmb{A}^T\pmb{Ax} - \pmb{A}^T \pmb{b}
        """
        return self._quadratic_cf.gradient(x)

    def hessian(self, x: NDArray[float64]) -> NDArray[float64]:
        r"""
        Hessian at x.

        .. math:: \pmb{A}^T\pmb{A}
        """
        return self._quadratic_cf.hessian(x)

    def proximal(self, x: NDArray[float64], penalty: float) -> NDArray[float64]:
        r"""
        Proximal at x.

        .. math::
            (\rho \pmb{A}^T \pmb{A} + \pmb{I})^{-1} (\pmb{x} + \rho \pmb{A}^T\pmb{b})

        This is a closed form solution, see :meth:`CostFunction.proximal` for the general proximal definition.
        """
        return self._quadratic_cf.proximal(x, penalty)
