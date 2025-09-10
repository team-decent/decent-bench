from __future__ import annotations

import copy
from abc import ABC, abstractmethod

import numpy as np
from numpy import float64
from numpy.typing import NDArray
from scipy import special


class CostFunction(ABC):
    """Used by agents to evaluate the cost and its derivatives at a certain x."""

    @property
    @abstractmethod
    def domain_shape(self) -> tuple[int, ...]:
        """Required shape of x."""

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

        If the cost function's proximal does not have a closed form solution, it can be solved iteratively using
        ``ProximalCost(self, y, rho).minimize()``.
        """

    @abstractmethod
    def __add__(self, other: CostFunction) -> CostFunction:
        """
        Add another cost function to create a new one.

        :class:`SumCost` can be used as the result of :meth:`__add__` by returning ``SumCost([self, other])``. However,
        it's often more efficient to preserve the cost function type if possible. For example, the addition of two
        :class:`QuadraticCost` objects benefits from returning a new :class:`QuadraticCost` instead of a
        :class:`SumCost` as this preserves the closed form proximal solution and only requires one evaluation instead of
        two when calling :meth:`evaluate`, :meth:`gradient`, and :meth:`hessian`.
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
        self.A_sym = 0.5 * (A + A.T)
        self.b = b
        self.c = c

    @property
    def domain_shape(self) -> tuple[int, ...]:  # noqa: D102
        return self.b.shape

    @property
    def m_smooth(self) -> float:
        r"""
        The cost function's smoothness constant.

        .. math::
            \max_{i} \left| \lambda_i \right|

        where :math:`\lambda_i` are the eigenvalues of :math:`\frac{1}{2} (\mathbf{A}+\mathbf{A}^T)`.

        For the general definition, see :attr:`CostFunction.m_smooth`.
        """
        eigs = np.linalg.eigvalsh(self.A_sym)
        return float(np.max(np.abs(eigs)))

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

        where :math:`\lambda_i` are the eigenvalues of :math:`\frac{1}{2} (\mathbf{A}+\mathbf{A}^T)`.

        For the general definition, see :attr:`CostFunction.m_cvx`.
        """
        eigs = np.linalg.eigvalsh(self.A_sym)
        l_min = float(np.min(eigs))
        if l_min > 0:
            return l_min
        if l_min == 0:
            return 0
        return np.nan

    def evaluate(self, x: NDArray[float64]) -> float:
        r"""
        Evaluate function at x.

        .. math:: \frac{1}{2} \mathbf{x}^T \mathbf{Ax} + \mathbf{b}^T \mathbf{x} + c
        """
        return float(0.5 * x.dot(self.A.dot(x)) + self.b.dot(x) + self.c)

    def gradient(self, x: NDArray[float64]) -> NDArray[float64]:
        r"""
        Gradient at x.

        .. math:: \frac{1}{2} (\mathbf{A}+\mathbf{A}^T)\mathbf{x} + \mathbf{b}
        """
        return self.A_sym @ x + self.b

    def hessian(self, x: NDArray[float64]) -> NDArray[float64]:  # noqa: ARG002
        r"""
        Hessian at x.

        .. math:: \frac{1}{2} (\mathbf{A}+\mathbf{A}^T)
        """
        return self.A_sym

    def proximal(self, y: NDArray[float64], rho: float) -> NDArray[float64]:
        r"""
        Proximal at y.

        .. math::
            (\frac{\rho}{2} (\mathbf{A} + \mathbf{A}^T) + \mathbf{I})^{-1} (\mathbf{y} - \rho \mathbf{b})

        where :math:`\rho > 0` is the penalty.

        This is a closed form solution, see :meth:`CostFunction.proximal` for the general proximal definition.
        """
        lhs = rho * self.A_sym + np.eye(self.A.shape[1])
        rhs = y - self.b * rho
        return np.asarray(np.linalg.solve(lhs, rhs), dtype=np.float64)

    def __add__(self, other: CostFunction) -> CostFunction:
        """
        Add another cost function.

        Raises:
            ValueError: if the domain shapes don't match

        """
        if self.domain_shape != other.domain_shape:
            raise ValueError(f"Mismatching domain shapes: {self.domain_shape} vs {other.domain_shape}")
        if isinstance(other, QuadraticCost):
            return QuadraticCost(self.A + other.A, self.b + other.b, self.c + other.c)
        if isinstance(other, (LinearRegressionCost, ProximalCost)):
            return self + other.inner
        return SumCost([self, other])


class LinearRegressionCost(CostFunction):
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
        self.inner = QuadraticCost(A.T.dot(A), -A.T.dot(b), 0.5 * b.dot(b))
        self.A = A
        self.b = b

    @property
    def domain_shape(self) -> tuple[int, ...]:  # noqa: D102
        return self.inner.domain_shape

    @property
    def m_smooth(self) -> float:
        r"""
        The cost function's smoothness constant.

        .. math::
            \max_{i} \left| \lambda_i \right|

        where :math:`\lambda_i` are the eigenvalues of :math:`\mathbf{A}^T \mathbf{A}`.

        For the general definition, see :attr:`CostFunction.m_smooth`.
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

        For the general definition, see :attr:`CostFunction.m_cvx`.
        """
        return self.inner.m_cvx

    def evaluate(self, x: NDArray[float64]) -> float:
        r"""
        Evaluate function at x.

        .. math:: \frac{1}{2} \| \mathbf{Ax} - \mathbf{b} \|^2
        """
        return self.inner.evaluate(x)

    def gradient(self, x: NDArray[float64]) -> NDArray[float64]:
        r"""
        Gradient at x.

        .. math:: \mathbf{A}^T\mathbf{Ax} - \mathbf{A}^T \mathbf{b}
        """
        return self.inner.gradient(x)

    def hessian(self, x: NDArray[float64]) -> NDArray[float64]:
        r"""
        Hessian at x.

        .. math:: \mathbf{A}^T\mathbf{A}
        """
        return self.inner.hessian(x)

    def proximal(self, y: NDArray[float64], rho: float) -> NDArray[float64]:
        r"""
        Proximal at y.

        .. math::
            (\rho \mathbf{A}^T \mathbf{A} + \mathbf{I})^{-1} (\mathbf{y} + \rho \mathbf{A}^T\mathbf{b})

        where :math:`\rho > 0` is the penalty.

        This is a closed form solution, see :meth:`CostFunction.proximal` for the general proximal definition.
        """
        return self.inner.proximal(y, rho)

    def __add__(self, other: CostFunction) -> CostFunction:
        """Add another cost function."""
        return self.inner + other


class LogisticRegressionCost(CostFunction):
    r"""
    Logistic regression cost function.

    .. math:: f(\mathbf{x}) =
        -\left[ \mathbf{b}^T \log( \sigma(\mathbf{Ax}) )
        + ( \mathbf{1} - \mathbf{b} )^T
            \log( 1 - \sigma(\mathbf{Ax}) ) \right]
    """

    def __init__(self, A: NDArray[float64], b: NDArray[float64]):  # noqa: N803
        if A.ndim != 2:
            raise ValueError("Matrix A must be 2D")
        if b.ndim != 1:
            raise ValueError("Vector b must be 1D")
        if A.shape[0] != b.shape[0]:
            raise ValueError(f"Dimension mismatch: A has shape {A.shape} but b has length {b.shape[0]}")
        class_labels = np.unique(b)
        if class_labels.shape != (2,):
            raise ValueError("Vector b must contain exactly two classes")
        b = copy.deepcopy(b)
        b[np.where(b == class_labels[0])], b[np.where(b == class_labels[1])] = 0, 1
        self.A = A
        self.b = b

    @property
    def domain_shape(self) -> tuple[int, ...]:  # noqa: D102
        return (self.A.shape[1],)

    @property
    def m_smooth(self) -> float:
        r"""
        The cost function's smoothness constant.

        .. math:: \frac{1}{4} \max_i \sum_{j} (\mathbf{A}_{ij})^2

        For the general definition, see :attr:`CostFunction.m_smooth`.
        """
        return np.max(np.sum(self.A**2, axis=1, dtype=float64)) / 4

    @property
    def m_cvx(self) -> float:
        """
        The cost function's convexity constant, 0.

        For the general definition, see :attr:`CostFunction.m_cvx`.
        """
        return 0

    def evaluate(self, x: NDArray[float64]) -> float:
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

    def gradient(self, x: NDArray[float64]) -> NDArray[float64]:
        r"""
        Gradient at x.

        .. math:: \mathbf{A}^T (\sigma(\mathbf{Ax}) - \mathbf{b})
        """
        sig = special.expit(self.A.dot(x))
        return self.A.T.dot(sig - self.b)  # type: ignore[no-any-return]

    def hessian(self, x: NDArray[float64]) -> NDArray[float64]:
        r"""
        Hessian at x.

        .. math:: \mathbf{A}^T \mathbf{DA}

        where :math:`\mathbf{D}` is a diagonal matrix such that
        :math:`\mathbf{D}_i = \sigma(\mathbf{Ax}_i) (1-\sigma(\mathbf{Ax}_i))`
        """
        sig = special.expit(self.A.dot(x))
        D = np.diag(sig * (1 - sig))  # noqa: N806
        return self.A.T.dot(D).dot(self.A)  # type: ignore[no-any-return]

    def proximal(self, y: NDArray[float64], rho: float) -> NDArray[float64]:
        """
        Proximal at y solved using an iterative method.

        See :meth:`CostFunction.proximal` for the general proximal definition.
        """
        return ProximalCost(self, y, rho).minimize()

    def __add__(self, other: CostFunction) -> CostFunction:
        """
        Add another cost function.

        Raises:
            ValueError: if the domain shapes don't match

        """
        if self.domain_shape != other.domain_shape:
            raise ValueError(f"Mismatching domain shapes: {self.domain_shape} vs {other.domain_shape}")
        if isinstance(other, LogisticRegressionCost):
            return LogisticRegressionCost(np.vstack([self.A, other.A]), np.concatenate([self.b, other.b]))
        if isinstance(other, ProximalCost):
            return self + other.inner
        return SumCost([self, other])


class SumCost(CostFunction):
    """The sum of multiple cost functions."""

    def __init__(self, cost_functions: list[CostFunction]):
        if not all(cost_functions[0].domain_shape == cf.domain_shape for cf in cost_functions):
            raise ValueError("All cost functions must have the same domain shape")
        self.cost_functions: list[CostFunction] = []
        for cf in cost_functions:
            if isinstance(cf, SumCost):
                self.cost_functions.extend(cf.cost_functions)
            else:
                self.cost_functions.append(cf)

    @property
    def domain_shape(self) -> tuple[int, ...]:  # noqa: D102
        return self.cost_functions[0].domain_shape

    @property
    def m_smooth(self) -> float:
        r"""
        The cost function's smoothness constant.

        .. math::
            \sum f_{k_\text{m_smooth}}

        provided all :math:`f_{k_\text{m_smooth}}` are finite.
        If any :math:`f_{k_\text{m_smooth}} = \text{NaN}`,
        the result is :math:`\text{NaN}`.

        For the general definition, see :attr:`CostFunction.m_smooth`.
        """
        m_smooth_vals = [cf.m_smooth for cf in self.cost_functions]
        return np.nan if any(np.isnan(v) for v in m_smooth_vals) else sum(m_smooth_vals)

    @property
    def m_cvx(self) -> float:
        r"""
        The cost function's convexity constant.

        .. math::
            \sum f_{k_\text{m_cvx}}

        provided all :math:`f_{k_\text{m_cvx}}` are finite.
        If any :math:`f_{k_\text{m_cvx}} = \text{NaN}`,
        the result is :math:`\text{NaN}`.

        For the general definition, see :attr:`CostFunction.m_cvx`.
        """
        m_cvx_vals = [cf.m_cvx for cf in self.cost_functions]
        return np.nan if any(np.isnan(v) for v in m_cvx_vals) else sum(m_cvx_vals)

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
        return ProximalCost(self, y, rho).minimize()

    def __add__(self, other: CostFunction) -> SumCost:
        """Add another cost function."""
        return SumCost([self, other])


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
        self.y = y
        self.inner = f + QuadraticCost(A=np.eye(len(y)) / rho, b=-y / rho, c=y.dot(y) / (2 * rho))

    @property
    def domain_shape(self) -> tuple[int, ...]:  # noqa: D102
        return self.inner.domain_shape

    @property
    def m_smooth(self) -> float:
        r"""
        The cost function's smoothness constant.

        .. math:: f_\text{m_smooth} + \frac{1}{\rho}

        For the general definition, see :attr:`CostFunction.m_smooth`.
        """
        return self.inner.m_smooth

    @property
    def m_cvx(self) -> float:
        r"""
        The cost function's convexity constant.

        .. math:: f_\text{m_cvx} + \frac{1}{\rho}

        For the general definition, see :attr:`CostFunction.m_cvx`.
        """
        return self.inner.m_cvx

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
        return ProximalCost(self, y, rho).minimize()

    def minimize(self) -> NDArray[float64]:
        """
        Find x that minimizes the proximal cost function using accelerated gradient descent.

        This is the solution to the proximal operator described in :meth:`CostFunction.proximal`. Therefore,
        :meth:`ProximalCost.minimize` can be used to solve the proximal for cost functions lacking a closed
        form solution.
        """
        from decent_bench.library.core import centralized_algorithms as ca  # noqa: PLC0415

        return ca.accelerated_gradient_descent(self, self.y, max_iter=100, stop_tol=1e-10, max_tol=None)

    def __add__(self, other: CostFunction) -> CostFunction:
        """Add another cost function."""
        return self.inner + other
