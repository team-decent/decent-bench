from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any

import numpy as np
import numpy.linalg as la
from numpy import float64
from numpy.typing import NDArray
from scipy import special

import decent_bench.centralized_algorithms as ca
import decent_bench.utils.interoperability as iop
from decent_bench.utils.parameter import X
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks, SupportedXTypes


class Cost[Xtype: SupportedXTypes](ABC):
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
        """The framework used by this cost function."""

    @property
    @abstractmethod
    def device(self) -> SupportedDevices:
        """The device used by this cost function."""

    @property
    @abstractmethod
    def m_smooth(self) -> float:
        r"""
        Lipschitz constant of the cost function's gradient.

        The gradient's Lipschitz constant m_smooth is the smallest value such that
        :math:`\| \nabla f(\mathbf{x_1}) - \nabla f(\mathbf{x_2}) \| \leq \text{m_smooth}
        \cdot \|\mathbf{x_1} - \mathbf{x_2}\|`
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
        :math:`f(\mathbf{x_1}) \geq f(\mathbf{x_2})
        + \nabla f(\mathbf{x_2})^T (\mathbf{x_1} - \mathbf{x_2})
        + \frac{\text{m_cvx}}{2} \|\mathbf{x_1} - \mathbf{x_2}\|^2`
        for all :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`.

        Returns:
            - positive finite number if function is strongly convex
            - ``0`` if function is convex but not strongly convex
            - ``np.nan`` if function is not guaranteed to be convex

        """

    @abstractmethod
    def function(self, x: X[Xtype]) -> float:
        """Evaluate function at x."""

    def evaluate(self, x: X[Xtype]) -> float:
        """Alias for :meth:`function`."""
        return self.function(x)

    def loss(self, x: X[Xtype]) -> float:
        """Alias for :meth:`function`."""
        return self.function(x)

    def f(self, x: X[Xtype]) -> float:
        """Alias for :meth:`function`."""
        return self.function(x)

    @abstractmethod
    def gradient(self, x: X[Xtype]) -> X[Xtype]:
        """Gradient at x."""

    @abstractmethod
    def hessian(self, x: X[Xtype]) -> X[Xtype]:
        """Hessian at x."""

    @abstractmethod
    def proximal(self, y: X[Xtype], rho: float) -> X[Xtype]:
        r"""
        Proximal at y.

        The proximal operator is defined as:

        .. include:: snippets/proximal_operator.rst

        If the cost function's proximal does not have a closed form solution, it can be solved iteratively using
        :meth:`~decent_bench.centralized_algorithms.proximal_solver`.
        """

    @abstractmethod
    def __add__(self, other: Cost[Xtype]) -> Cost[Xtype]:
        """
        Add another cost function to create a new one.

        :class:`~SumCost` can be used as the result of :meth:`__add__` by returning
        ``SumCost([self, other])``. However, it's often more efficient to preserve the cost function type if possible.
        For example, the addition of two :class:`~QuadraticCost` objects benefits from returning a new
        :class:`~QuadraticCost` instead of a :class:`~SumCost` as this preserves the closed
        form proximal solution and only requires one evaluation instead of two when calling :meth:`evaluate`,
        :meth:`gradient`, and :meth:`hessian`.
        """


class QuadraticCost(Cost[NDArray[float64]]):
    r"""
    Quadratic cost function.

    .. math:: f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^T \mathbf{Ax} + \mathbf{b}^T \mathbf{x} + c
    """

    def __init__(self, A: X[NDArray[Any]], b: X[NDArray[Any]], c: float):  # noqa: N803
        if A.ndim != 2:
            raise ValueError("Matrix A must be 2D")
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix A must be square")
        if b.ndim != 1:
            raise ValueError("Vector b must be 1D")
        if A.shape[0] != b.shape[0]:
            raise ValueError(f"Dimension mismatch: A has shape {A.shape} but b has length {b.shape[0]}")
        self.A = A
        self.A_sym = 0.5 * (A + A.T)
        self.b = b
        self.c = c

    @property
    def shape(self) -> tuple[int, ...]:  # noqa: D102
        return self.b.shape

    @property
    def framework(self) -> SupportedFrameworks:  # noqa: D102
        return SupportedFrameworks.NUMPY

    @property
    def device(self) -> SupportedDevices:  # noqa: D102
        return SupportedDevices.CPU

    @cached_property
    def m_smooth(self) -> float:
        r"""
        The cost function's smoothness constant.

        .. math::
            \max_{i} \left| \lambda_i \right|

        where :math:`\lambda_i` are the eigenvalues of :math:`\frac{1}{2} (\mathbf{A}+\mathbf{A}^T)`.

        For the general definition, see
        :attr:`Cost.m_smooth <decent_bench.costs.Cost.m_smooth>`.
        """
        eigs = np.linalg.eigvalsh(self.A_sym)
        return float(np.max(np.abs(eigs)))

    @cached_property
    def m_cvx(self) -> float:
        r"""
        The cost function's convexity constant.

        .. math::
            \begin{array}{ll}
                \min_i \lambda_i, & \text{if } \min_i \lambda_i > 0, \\
                0, & \text{if } \min_i \lambda_i = 0, \\
                \text{NaN}, & \text{if } \min_i \lambda_i < 0
            \end{array}

        where :math:`\lambda_i` are the eigenvalues of :math:`\frac{1}{2} (\mathbf{A}+\mathbf{A}^T)`.

        For the general definition, see
        :attr:`Cost.m_cvx <decent_bench.costs.Cost.m_cvx>`.
        """
        eigs = np.linalg.eigvalsh(self.A_sym)
        l_min = float(np.min(eigs))
        tol = 1e-12
        if l_min > tol:
            return l_min
        if abs(l_min) <= tol:
            return 0
        return np.nan

    def function(self, x: X[NDArray[float64]]) -> float:
        r"""
        Evaluate function at x.

        .. math:: \frac{1}{2} \mathbf{x}^T \mathbf{Ax} + \mathbf{b}^T \mathbf{x} + c
        """
        return float(0.5 * x.dot(self.A.dot(x)) + self.b.dot(x) + self.c)

    def gradient(self, x: X[NDArray[float64]]) -> X[NDArray[float64]]:
        r"""
        Gradient at x.

        .. math:: \frac{1}{2} (\mathbf{A}+\mathbf{A}^T)\mathbf{x} + \mathbf{b}
        """
        return self.A_sym @ x + self.b

    def hessian(self, x: X[NDArray[float64]]) -> X[NDArray[float64]]:  # noqa: ARG002
        r"""
        Hessian at x.

        .. math:: \frac{1}{2} (\mathbf{A}+\mathbf{A}^T)
        """
        return self.A_sym

    def proximal(self, y: X[NDArray[float64]], rho: float) -> X[NDArray[float64]]:
        r"""
        Proximal at y.

        .. math::
            (\frac{\rho}{2} (\mathbf{A} + \mathbf{A}^T) + \mathbf{I})^{-1} (\mathbf{y} - \rho \mathbf{b})

        where :math:`\rho > 0` is the penalty.

        This is a closed form solution, see
        :meth:`Cost.proximal() <decent_bench.costs.Cost.proximal>`
        for the general proximal definition.
        """
        lhs = rho * self.A_sym + np.eye(self.A.shape[1])
        rhs = y - self.b * rho
        return iop.numpy_to_X(
            np.asarray(np.linalg.solve(lhs, rhs), dtype=float64),
            framework=self.framework,
            device=self.device,
        )

    def __add__(self, other: Cost[NDArray[float64]]) -> Cost[NDArray[float64]]:
        """
        Add another cost function.

        Raises:
            ValueError: if the domain shapes don't match

        """
        if self.shape != other.shape:
            raise ValueError(f"Mismatching domain shapes: {self.shape} vs {other.shape}")
        if isinstance(other, QuadraticCost):
            return QuadraticCost(self.A + other.A, self.b + other.b, self.c + other.c)
        if isinstance(other, LinearRegressionCost):
            return self + other.inner
        return SumCost([self, other])


class LinearRegressionCost(Cost[NDArray[float64]]):
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

    def __init__(self, A: X[NDArray[Any]], b: X[NDArray[Any]]):  # noqa: N803
        if A.shape[0] != b.shape[0]:
            raise ValueError(f"Dimension mismatch: A has {A.shape[0]} rows but b has {b.shape[0]} elements")
        self.inner = QuadraticCost(A.T.dot(A), -A.T.dot(b), 0.5 * b.dot(b).float)
        self.A = A
        self.b = b

    @property
    def shape(self) -> tuple[int, ...]:  # noqa: D102
        return self.inner.shape

    @property
    def framework(self) -> SupportedFrameworks:  # noqa: D102
        return SupportedFrameworks.NUMPY

    @property
    def device(self) -> SupportedDevices:  # noqa: D102
        return SupportedDevices.CPU

    @property
    def m_smooth(self) -> float:
        r"""
        The cost function's smoothness constant.

        .. math::
            \max_{i} \left| \lambda_i \right|

        where :math:`\lambda_i` are the eigenvalues of :math:`\mathbf{A}^T \mathbf{A}`.

        For the general definition, see
        :attr:`Cost.m_smooth <decent_bench.costs.Cost.m_smooth>`.
        """
        return self.inner.m_smooth

    @property
    def m_cvx(self) -> float:
        r"""
        The cost function's convexity constant.

        .. math::
            \begin{array}{ll}
                \min_i \lambda_i, & \text{if } \min_i \lambda_i > 0, \\
                0, & \text{if } \min_i \lambda_i = 0, \\
                \text{NaN}, & \text{if } \min_i \lambda_i < 0
            \end{array}

        where :math:`\lambda_i` are the eigenvalues of :math:`\mathbf{A}^T \mathbf{A}`.

        For the general definition, see
        :attr:`Cost.m_cvx <decent_bench.costs.Cost.m_cvx>`.
        """
        return self.inner.m_cvx

    def function(self, x: X[NDArray[float64]]) -> float:
        r"""
        Evaluate function at x.

        .. math:: \frac{1}{2} \| \mathbf{Ax} - \mathbf{b} \|^2
        """
        return self.inner.function(x)

    def gradient(self, x: X[NDArray[float64]]) -> X[NDArray[float64]]:
        r"""
        Gradient at x.

        .. math:: \mathbf{A}^T\mathbf{Ax} - \mathbf{A}^T \mathbf{b}
        """
        return self.inner.gradient(x)

    def hessian(self, x: X[NDArray[float64]]) -> X[NDArray[float64]]:
        r"""
        Hessian at x.

        .. math:: \mathbf{A}^T\mathbf{A}
        """
        return self.inner.hessian(x)

    def proximal(self, y: X[NDArray[float64]], rho: float) -> X[NDArray[float64]]:
        r"""
        Proximal at y.

        .. math::
            (\rho \mathbf{A}^T \mathbf{A} + \mathbf{I})^{-1} (\mathbf{y} + \rho \mathbf{A}^T\mathbf{b})

        where :math:`\rho > 0` is the penalty.

        This is a closed form solution, see
        :meth:`Cost.proximal() <decent_bench.costs.Cost.proximal>`
        for the general proximal definition.
        """
        return self.inner.proximal(y, rho)

    def __add__(self, other: Cost[NDArray[float64]]) -> Cost[NDArray[float64]]:
        """Add another cost function."""
        return self.inner + other


class LogisticRegressionCost(Cost[NDArray[float64]]):
    r"""
    Logistic regression cost function.

    .. math:: f(\mathbf{x}) =
        -\left[ \mathbf{b}^T \log( \sigma(\mathbf{Ax}) )
        + ( \mathbf{1} - \mathbf{b} )^T
            \log( 1 - \sigma(\mathbf{Ax}) ) \right]
    """

    def __init__(self, A: X[NDArray[Any]], b: X[NDArray[Any]]):  # noqa: N803
        if A.ndim != 2:
            raise ValueError("Matrix A must be 2D")
        if b.ndim != 1:
            raise ValueError("Vector b must be 1D")
        if A.shape[0] != b.shape[0]:
            raise ValueError(f"Dimension mismatch: A has shape {A.shape} but b has length {b.shape[0]}")
        class_labels = np.unique(b)
        if class_labels.shape != (2,):
            raise ValueError("Vector b must contain exactly two classes")

        b[np.where(b == class_labels[0])], b[np.where(b == class_labels[1])] = 0, 1
        self.A = A
        self.b = b

    @property
    def shape(self) -> tuple[int, ...]:  # noqa: D102
        return (self.A.shape[1],)

    @property
    def framework(self) -> SupportedFrameworks:  # noqa: D102
        return SupportedFrameworks.NUMPY

    @property
    def device(self) -> SupportedDevices:  # noqa: D102
        return SupportedDevices.CPU

    @cached_property
    def m_smooth(self) -> float:
        r"""
        The cost function's smoothness constant.

        .. math:: \frac{m}{4} \max_i \|\mathbf{A}_i\|^2

        where m is the number of rows in :math:`\mathbf{A}`.

        For the general definition, see
        :attr:`Cost.m_smooth <decent_bench.costs.Cost.m_smooth>`.
        """
        return float(max(pow(la.norm(row), 2) for row in self.A) * self.A.shape[0] / 4)

    @property
    def m_cvx(self) -> float:
        """
        The cost function's convexity constant, 0.

        For the general definition, see
        :attr:`Cost.m_cvx <decent_bench.costs.Cost.m_cvx>`.
        """
        return 0

    def function(self, x: X[NDArray[float64]]) -> float:
        r"""
        Evaluate function at x.

        .. math::
            -\left[ \mathbf{b}^T \log( \sigma(\mathbf{Ax}) )
            + ( \mathbf{1} - \mathbf{b} )^T
                \log( 1 - \sigma(\mathbf{Ax}) ) \right]
        """
        Ax = self.A.dot(x)  # noqa: N806
        neg_log_sig = np.logaddexp(0.0, -Ax)
        cost = self.b.dot(neg_log_sig) + (1 - self.b).dot(Ax + neg_log_sig)
        return float(cost)

    def gradient(self, x: X[NDArray[float64]]) -> X[NDArray[float64]]:
        r"""
        Gradient at x.

        .. math:: \mathbf{A}^T (\sigma(\mathbf{Ax}) - \mathbf{b})
        """
        sig = special.expit(self.A.dot(x))
        return self.A.T.dot(sig - self.b)

    def hessian(self, x: X[NDArray[float64]]) -> X[NDArray[float64]]:
        r"""
        Hessian at x.

        .. math:: \mathbf{A}^T \mathbf{DA}

        where :math:`\mathbf{D}` is a diagonal matrix such that
        :math:`\mathbf{D}_i = \sigma(\mathbf{Ax}_i) (1-\sigma(\mathbf{Ax}_i))`
        """
        sig = special.expit(self.A.dot(x))
        D = np.diag(sig * (1 - sig))  # noqa: N806
        return self.A.T.dot(D).dot(self.A)

    def proximal(self, y: X[NDArray[float64]], rho: float) -> X[NDArray[float64]]:
        """
        Proximal at y solved using an iterative method.

        See
        :meth:`Cost.proximal() <decent_bench.costs.Cost.proximal>`
        for the general proximal definition.
        """
        return ca.proximal_solver(self, y, rho)

    def __add__(self, other: Cost[NDArray[float64]]) -> Cost[NDArray[float64]]:
        """
        Add another cost function.

        Raises:
            ValueError: if the domain shapes don't match

        """
        if self.shape != other.shape:
            raise ValueError(f"Mismatching domain shapes: {self.shape} vs {other.shape}")
        if isinstance(other, LogisticRegressionCost):
            return LogisticRegressionCost(
                iop.numpy_to_X(np.vstack([self.A, other.A]), framework=self.framework, device=self.device),
                iop.numpy_to_X(np.concatenate([self.b, other.b]), framework=self.framework, device=self.device),
            )
        return SumCost([self, other])


class SumCost(Cost[NDArray[float64]]):
    """The sum of multiple cost functions."""

    def __init__(self, costs: list[Cost[NDArray[float64]]]):
        if not all(costs[0].shape == cf.shape for cf in costs):
            raise ValueError("All cost functions must have the same domain shape")
        self.costs: list[Cost[NDArray[float64]]] = []
        for cf in costs:
            if isinstance(cf, SumCost):
                self.costs.extend(cf.costs)
            else:
                self.costs.append(cf)

    @property
    def shape(self) -> tuple[int, ...]:  # noqa: D102
        return self.costs[0].shape

    @property
    def framework(self) -> SupportedFrameworks:  # noqa: D102
        return self.costs[0].framework

    @property
    def device(self) -> SupportedDevices:  # noqa: D102
        return self.costs[0].device

    @cached_property
    def m_smooth(self) -> float:
        r"""
        The cost function's smoothness constant.

        .. math::
            \sum f_{k_\text{m_smooth}}

        provided all :math:`f_{k_\text{m_smooth}}` are finite.
        If any :math:`f_{k_\text{m_smooth}} = \text{NaN}`,
        the result is :math:`\text{NaN}`.

        For the general definition, see
        :attr:`Cost.m_smooth <decent_bench.costs.Cost.m_smooth>`.
        """
        m_smooth_vals = [cf.m_smooth for cf in self.costs]
        return np.nan if any(np.isnan(v) for v in m_smooth_vals) else sum(m_smooth_vals)

    @cached_property
    def m_cvx(self) -> float:
        r"""
        The cost function's convexity constant.

        .. math::
            \sum f_{k_\text{m_cvx}}

        provided all :math:`f_{k_\text{m_cvx}}` are finite.
        If any :math:`f_{k_\text{m_cvx}} = \text{NaN}`,
        the result is :math:`\text{NaN}`.

        For the general definition, see
        :attr:`Cost.m_cvx <decent_bench.costs.Cost.m_cvx>`.
        """
        m_cvx_vals = [cf.m_cvx for cf in self.costs]
        return np.nan if any(np.isnan(v) for v in m_cvx_vals) else sum(m_cvx_vals)

    def function(self, x: X[NDArray[float64]]) -> float:
        """Sum the :meth:`function` of each cost function."""
        return sum(cf.function(x) for cf in self.costs)

    def gradient(self, x: X[NDArray[float64]]) -> X[NDArray[float64]]:
        """Sum the :meth:`gradient` of each cost function."""
        return iop.sum(iop.stack([cf.gradient(x) for cf in self.costs]), dim=0)

    def hessian(self, x: X[NDArray[float64]]) -> X[NDArray[float64]]:
        """Sum the :meth:`hessian` of each cost function."""
        return iop.sum(iop.stack([cf.hessian(x) for cf in self.costs]), dim=0)

    def proximal(self, y: X[NDArray[float64]], rho: float) -> X[NDArray[float64]]:
        """
        Proximal at y solved using an iterative method.

        See
        :meth:`Cost.proximal() <decent_bench.costs.Cost.proximal>`
        for the general proximal definition.
        """
        return ca.proximal_solver(self, y, rho)

    def __add__(self, other: Cost[NDArray[float64]]) -> SumCost:
        """Add another cost function."""
        return SumCost([self, other])
