import copy

import numpy as np
from numpy import float64
from numpy.typing import NDArray
from scipy import special

import decent_bench.library.core.cent_algorithms as ca
from decent_bench.library.core.cost_functions.cost_function import CostFunction
from decent_bench.library.core.cost_functions.sum_cost import SumCost


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

        For the general definition, see
        :attr:`CostFunction.m_smooth <decent_bench.library.core.cost_functions.cost_function.CostFunction.m_smooth>`.
        """
        return np.max(np.sum(self.A**2, axis=1, dtype=float64)) / 4

    @property
    def m_cvx(self) -> float:
        """
        The cost function's convexity constant, 0.

        For the general definition, see
        :attr:`CostFunction.m_cvx <decent_bench.library.core.cost_functions.cost_function.CostFunction.m_cvx>`.
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
        res: NDArray[float64] = self.A.T.dot(sig - self.b)
        return res

    def hessian(self, x: NDArray[float64]) -> NDArray[float64]:
        r"""
        Hessian at x.

        .. math:: \mathbf{A}^T \mathbf{DA}

        where :math:`\mathbf{D}` is a diagonal matrix such that
        :math:`\mathbf{D}_i = \sigma(\mathbf{Ax}_i) (1-\sigma(\mathbf{Ax}_i))`
        """
        sig = special.expit(self.A.dot(x))
        D = np.diag(sig * (1 - sig))  # noqa: N806
        res: NDArray[float64] = self.A.T.dot(D).dot(self.A)
        return res

    def proximal(self, y: NDArray[float64], rho: float) -> NDArray[float64]:
        """
        Proximal at y solved using an iterative method.

        See
        :meth:`CostFunction.proximal() <decent_bench.library.core.cost_functions.cost_function.CostFunction.proximal>`
        for the general proximal definition.
        """
        return ca.proximal_solver(self, y, rho)

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
        return SumCost([self, other])
