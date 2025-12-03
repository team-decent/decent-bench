from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np
import numpy.linalg as la
from numpy import float64
from scipy import special

import decent_bench.centralized_algorithms as ca
import decent_bench.utils.interoperability as iop
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
    def function(self, x: Array) -> float:
        """Evaluate function at x."""

    def evaluate(self, x: Array) -> float:
        """Alias for :meth:`function`."""
        return self.function(x)

    def loss(self, x: Array) -> float:
        """Alias for :meth:`function`."""
        return self.function(x)

    def f(self, x: Array) -> float:
        """Alias for :meth:`function`."""
        return self.function(x)

    @abstractmethod
    def gradient(self, x: Array) -> Array:
        """Gradient at x."""

    @abstractmethod
    def hessian(self, x: Array) -> Array:
        """Hessian at x."""

    @abstractmethod
    def proximal(self, y: Array, rho: float) -> Array:
        r"""
        Proximal at y.

        The proximal operator is defined as:

        .. include:: snippets/proximal_operator.rst

        If the cost function's proximal does not have a closed form solution, it can be solved iteratively using
        :meth:`~decent_bench.centralized_algorithms.proximal_solver`.
        """

    @abstractmethod
    def __add__(self, other: Cost) -> Cost:
        """
        Add another cost function to create a new one.

        :class:`~SumCost` can be used as the result of :meth:`__add__` by returning
        ``SumCost([self, other])``. However, it's often more efficient to preserve the cost function type if possible.
        For example, the addition of two :class:`~QuadraticCost` objects benefits from returning a new
        :class:`~QuadraticCost` instead of a :class:`~SumCost` as this preserves the closed
        form proximal solution and only requires one evaluation instead of two when calling :meth:`evaluate`,
        :meth:`gradient`, and :meth:`hessian`.
        """


class QuadraticCost(Cost):
    r"""
    Quadratic cost function.

    .. math:: f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^T \mathbf{Ax} + \mathbf{b}^T \mathbf{x} + c
    """

    def __init__(self, A: Array, b: Array, c: float):  # noqa: N803
        self.A = iop.to_numpy(A)
        self.b = iop.to_numpy(b)

        if self.A.ndim != 2:
            raise ValueError("Matrix A must be 2D")
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError("Matrix A must be square")
        if self.b.ndim != 1:
            raise ValueError("Vector b must be 1D")
        if self.A.shape[0] != self.b.shape[0]:
            raise ValueError(f"Dimension mismatch: A has shape {self.A.shape} but b has length {self.b.shape[0]}")

        self.A_sym = 0.5 * (self.A + self.A.T)
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
    def m_smooth(self) -> float:  # pyright: ignore[reportIncompatibleMethodOverride]
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
    def m_cvx(self) -> float:  # pyright: ignore[reportIncompatibleMethodOverride]
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

    def function(self, x: Array) -> float:
        r"""
        Evaluate function at x.

        .. math:: \frac{1}{2} \mathbf{x}^T \mathbf{Ax} + \mathbf{b}^T \mathbf{x} + c
        """
        x_np = iop.to_numpy(x)
        return float(0.5 * x_np.dot(self.A.dot(x_np)) + self.b.dot(x_np) + self.c)

    def gradient(self, x: Array) -> Array:
        r"""
        Gradient at x.

        .. math:: \frac{1}{2} (\mathbf{A}+\mathbf{A}^T)\mathbf{x} + \mathbf{b}
        """
        ret: Array = self.A_sym @ x + self.b
        return ret

    def hessian(self, x: Array) -> Array:  # noqa: ARG002
        r"""
        Hessian at x.

        .. math:: \frac{1}{2} (\mathbf{A}+\mathbf{A}^T)
        """
        return iop.to_array(self.A_sym)  # Using iop.to_array because self.A_sym is a numpy array

    def proximal(self, y: Array, rho: float) -> Array:
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
        rhs = iop.to_numpy(y) - self.b * rho

        # Using iop.to_array because the result is a numpy array
        return iop.to_array(np.asarray(np.linalg.solve(lhs, rhs), dtype=float64))

    def __add__(self, other: Cost) -> Cost:
        """
        Add another cost function.

        Raises:
            ValueError: if the domain shapes don't match

        """
        if self.shape != other.shape:
            raise ValueError(f"Mismatching domain shapes: {self.shape} vs {other.shape}")
        if isinstance(other, QuadraticCost):
            # Using iop.to_array because the result is a numpy array
            return QuadraticCost(iop.to_array(self.A + other.A), iop.to_array(self.b + other.b), self.c + other.c)
        if isinstance(other, LinearRegressionCost):
            return self + other.inner
        return SumCost([self, other])


class LinearRegressionCost(Cost):
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

    def __init__(self, A: Array, b: Array):  # noqa: N803
        if iop.shape(A)[0] != iop.shape(b)[0]:
            raise ValueError(f"Dimension mismatch: A has {iop.shape(A)[0]} rows but b has {iop.shape(b)[0]} elements")
        self.inner = QuadraticCost(
            iop.dot(iop.transpose(A), A), -iop.dot(iop.transpose(A), b), float(0.5 * iop.dot(b, b))
        )
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

    def function(self, x: Array) -> float:
        r"""
        Evaluate function at x.

        .. math:: \frac{1}{2} \| \mathbf{Ax} - \mathbf{b} \|^2
        """
        return self.inner.function(x)

    def gradient(self, x: Array) -> Array:
        r"""
        Gradient at x.

        .. math:: \mathbf{A}^T\mathbf{Ax} - \mathbf{A}^T \mathbf{b}
        """
        return self.inner.gradient(x)

    def hessian(self, x: Array) -> Array:
        r"""
        Hessian at x.

        .. math:: \mathbf{A}^T\mathbf{A}
        """
        return self.inner.hessian(x)

    def proximal(self, y: Array, rho: float) -> Array:
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

    def __add__(self, other: Cost) -> Cost:
        """Add another cost function."""
        return self.inner + other


class LogisticRegressionCost(Cost):
    r"""
    Logistic regression cost function.

    .. math:: f(\mathbf{x}) =
        -\left[ \mathbf{b}^T \log( \sigma(\mathbf{Ax}) )
        + ( \mathbf{1} - \mathbf{b} )^T
            \log( 1 - \sigma(\mathbf{Ax}) ) \right]
    """

    def __init__(self, A: Array, b: Array):  # noqa: N803
        if len(iop.shape(A)) != 2:
            raise ValueError("Matrix A must be 2D")
        if len(iop.shape(b)) != 1:
            raise ValueError("Vector b must be 1D")
        if iop.shape(A)[0] != iop.shape(b)[0]:
            raise ValueError(f"Dimension mismatch: A has shape {iop.shape(A)} but b has length {iop.shape(b)[0]}")
        class_labels = np.unique(iop.to_numpy(b))
        if class_labels.shape != (2,):
            raise ValueError("Vector b must contain exactly two classes")

        b[np.where(b == class_labels[0])], b[np.where(b == class_labels[1])] = 0, 1
        self.A = iop.to_numpy(A)
        self.b = iop.to_numpy(b)

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
    def m_smooth(self) -> float:  # pyright: ignore[reportIncompatibleMethodOverride]
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

    def function(self, x: Array) -> float:
        r"""
        Evaluate function at x.

        .. math::
            -\left[ \mathbf{b}^T \log( \sigma(\mathbf{Ax}) )
            + ( \mathbf{1} - \mathbf{b} )^T
                \log( 1 - \sigma(\mathbf{Ax}) ) \right]
        """
        Ax = self.A.dot(iop.to_numpy(x))  # noqa: N806
        neg_log_sig = np.logaddexp(0.0, -Ax)
        cost = self.b.dot(neg_log_sig) + (1 - self.b).dot(Ax + neg_log_sig)
        return float(cost)

    def gradient(self, x: Array) -> Array:
        r"""
        Gradient at x.

        .. math:: \mathbf{A}^T (\sigma(\mathbf{Ax}) - \mathbf{b})
        """
        sig = special.expit(self.A.dot(iop.to_numpy(x)))
        return iop.to_array(self.A.T.dot(sig - self.b))

    def hessian(self, x: Array) -> Array:
        r"""
        Hessian at x.

        .. math:: \mathbf{A}^T \mathbf{DA}

        where :math:`\mathbf{D}` is a diagonal matrix such that
        :math:`\mathbf{D}_i = \sigma(\mathbf{Ax}_i) (1-\sigma(\mathbf{Ax}_i))`
        """
        sig = special.expit(self.A.dot(iop.to_numpy(x)))
        D = np.diag(sig * (1 - sig))  # noqa: N806
        return iop.to_array(self.A.T.dot(D).dot(self.A))

    def proximal(self, y: Array, rho: float) -> Array:
        """
        Proximal at y solved using an iterative method.

        See
        :meth:`Cost.proximal() <decent_bench.costs.Cost.proximal>`
        for the general proximal definition.
        """
        return ca.proximal_solver(self, y, rho)

    def __add__(self, other: Cost) -> Cost:
        """
        Add another cost function.

        Raises:
            ValueError: if the domain shapes don't match

        """
        if self.shape != other.shape:
            raise ValueError(f"Mismatching domain shapes: {self.shape} vs {other.shape}")
        if isinstance(other, LogisticRegressionCost):
            return LogisticRegressionCost(
                iop.to_array(np.vstack([self.A, other.A])),
                iop.to_array(np.concatenate([self.b, other.b])),
            )
        return SumCost([self, other])


class SumCost(Cost):
    """The sum of multiple cost functions."""

    def __init__(self, costs: list[Cost]):
        if not all(costs[0].shape == cf.shape for cf in costs):
            raise ValueError("All cost functions must have the same domain shape")
        self.costs: list[Cost] = []
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
    def m_smooth(self) -> float:  # pyright: ignore[reportIncompatibleMethodOverride]
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
    def m_cvx(self) -> float:  # pyright: ignore[reportIncompatibleMethodOverride]
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

    def function(self, x: Array) -> float:
        """Sum the :meth:`function` of each cost function."""
        return sum(cf.function(x) for cf in self.costs)

    def gradient(self, x: Array) -> Array:
        """Sum the :meth:`gradient` of each cost function."""
        return iop.sum(iop.stack([cf.gradient(x) for cf in self.costs]), dim=0)

    def hessian(self, x: Array) -> Array:
        """Sum the :meth:`hessian` of each cost function."""
        return iop.sum(iop.stack([cf.hessian(x) for cf in self.costs]), dim=0)

    def proximal(self, y: Array, rho: float) -> Array:
        """
        Proximal at y solved using an iterative method.

        See
        :meth:`Cost.proximal() <decent_bench.costs.Cost.proximal>`
        for the general proximal definition.
        """
        return ca.proximal_solver(self, y, rho)

    def __add__(self, other: Cost) -> SumCost:
        """Add another cost function."""
        return SumCost([self, other])
