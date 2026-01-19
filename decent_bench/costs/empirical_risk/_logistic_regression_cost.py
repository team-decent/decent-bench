from __future__ import annotations

from functools import cached_property
from typing import Any

import numpy as np
import numpy.linalg as la
from numpy import float64
from numpy.typing import NDArray
from scipy import special

import decent_bench.centralized_algorithms as ca
import decent_bench.utils.interoperability as iop
from decent_bench.costs.base._sum_cost import SumCost
from decent_bench.costs.base._cost import Cost
from decent_bench.datasets import DatasetPartition
from decent_bench.utils.array import Array
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks

from ._empirical_risk_cost import EmpiricalRiskCost


class LogisticRegressionCost(EmpiricalRiskCost):
    r"""
    Logistic regression cost function.

    .. math:: f(\mathbf{x}) =
        -\left[ \mathbf{b}^T \log( \sigma(\mathbf{Ax}) )
        + ( \mathbf{1} - \mathbf{b} )^T
            \log( 1 - \sigma(\mathbf{Ax}) ) \right]
    """

    def __init__(self, A: Array, b: Array, batch_size: int | None = None, batch_seed: int | None = None):  # noqa: N803
        if len(iop.shape(A)) != 2:
            raise ValueError("Matrix A must be 2D")
        if len(iop.shape(b)) != 1:
            raise ValueError("Vector b must be 1D")
        if iop.shape(A)[0] != iop.shape(b)[0]:
            raise ValueError(f"Dimension mismatch: A has shape {iop.shape(A)} but b has length {iop.shape(b)[0]}")
        class_labels = np.unique(iop.to_numpy(b))
        if class_labels.shape != (2,):
            raise ValueError("Vector b must contain exactly two classes")

        self.A: NDArray[float64] = iop.to_numpy(A)
        self.b: NDArray[float64] = iop.to_numpy(iop.copy(b))  # Copy b to avoid modifying original array pointer
        (
            self.b[np.where(self.b == class_labels[0])],
            self.b[np.where(self.b == class_labels[1])],
        ) = (0, 1)
        self._batch_size = batch_size if batch_size is not None else self.A.shape[0]
        self._last_batch_used: list[int] = []
        self._rand = np.random.default_rng(seed=batch_seed)

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.A.shape[1],)

    @property
    def framework(self) -> SupportedFrameworks:
        return SupportedFrameworks.NUMPY

    @property
    def device(self) -> SupportedDevices:
        return SupportedDevices.CPU

    @property
    def n_samples(self) -> int:
        return self.A.shape[0]

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def batch_used(self) -> list[int]:
        return self._last_batch_used

    @cached_property
    def m_smooth(self) -> float:  # pyright: ignore[reportIncompatibleMethodOverride]
        r"""
        The cost function's smoothness constant.

        .. math:: \frac{m}{4} \max_i \|\mathbf{A}_i\|^2

        where m is the number of rows in :math:`\mathbf{A}`.

        For the general definition, see
        :attr:`Cost.m_smooth <decent_bench.costs.base.Cost.m_smooth>`.
        """
        return float(max(pow(la.norm(row), 2) for row in self.A) * self.A.shape[0] / 4)

    @property
    def m_cvx(self) -> float:
        """
        The cost function's convexity constant, 0.

        For the general definition, see
        :attr:`Cost.m_cvx <decent_bench.costs.base.Cost.m_cvx>`.
        """
        return 0

    @iop.autodecorate_cost_method(EmpiricalRiskCost.predict)
    def predict(self, x: NDArray[float64], data: list[NDArray[float64]]) -> NDArray[float64]:
        """
        Make predictions at x on the given data.

        Args:
            x: Point to make predictions at.
            data: List of NDArray containing data to make predictions on.

        Returns:
            Predictions as a binary array.

        """
        pred_data = np.stack(data) if isinstance(data, list) else data
        logits = pred_data.dot(x)
        sig = special.expit(logits)
        return (sig >= 0.5).astype(float)

    @iop.autodecorate_cost_method(EmpiricalRiskCost.function)
    def function(self, x: NDArray[float64], indices: list[int] | None = None) -> float:
        r"""
        Evaluate function at x.

        .. math::
            -\left[ \mathbf{b}^T \log( \sigma(\mathbf{Ax}) )
            + ( \mathbf{1} - \mathbf{b} )^T
                \log( 1 - \sigma(\mathbf{Ax}) ) \right]
        """
        A, b = self._get_batch_data(indices)  # noqa: N806
        Ax = A.dot(x)  # noqa: N806
        neg_log_sig = np.logaddexp(0.0, -Ax)
        cost = b.dot(neg_log_sig) + (1 - b).dot(Ax + neg_log_sig)
        return float(cost)

    @iop.autodecorate_cost_method(EmpiricalRiskCost.gradient)
    def gradient(self, x: NDArray[float64], indices: list[int] | None = None) -> NDArray[float64]:
        r"""
        Gradient at x.

        .. math:: \mathbf{A}^T (\sigma(\mathbf{Ax}) - \mathbf{b})
        """
        A, b = self._get_batch_data(indices)  # noqa: N806
        sig = special.expit(A.dot(x))
        res: NDArray[float64] = A.T.dot(sig - b)
        return res

    @iop.autodecorate_cost_method(EmpiricalRiskCost.hessian)
    def hessian(self, x: NDArray[float64], indices: list[int] | None = None) -> NDArray[float64]:
        r"""
        Hessian at x.

        .. math:: \mathbf{A}^T \mathbf{DA}

        where :math:`\mathbf{D}` is a diagonal matrix such that
        :math:`\mathbf{D}_i = \sigma(\mathbf{Ax}_i) (1-\sigma(\mathbf{Ax}_i))`
        """
        A, _ = self._get_batch_data(indices)  # noqa: N806
        sig = special.expit(A.dot(x))
        D = np.diag(sig * (1 - sig))  # noqa: N806
        res: NDArray[float64] = A.T.dot(D).dot(A)
        return res

    def proximal(self, x: Array, rho: float, **kwargs: Any) -> Array:  # noqa: ANN401, ARG002
        """
        Proximal at x solved using an iterative method.

        See
        :meth:`Cost.proximal() <decent_bench.costs.base.Cost.proximal>`
        for the general proximal definition.
        """
        return ca.proximal_solver(self, x, rho)

    def _get_batch_data(self, indices: list[int] | None) -> tuple[NDArray[float64], NDArray[float64]]:
        """Get data for a batch. Returns A and b for the batch."""
        if indices is not None:
            self._last_batch_used = indices
        elif self.batch_size < self.n_samples:
            # Sample a random batch
            self._last_batch_used = self._rand.choice(self.n_samples, size=self._batch_size, replace=False).tolist()
        else:
            # Use full dataset
            self._last_batch_used = list(range(self.n_samples))
            return self.A, self.b

        return self.A[self._last_batch_used, :], self.b[self._last_batch_used]

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
                iop.to_array(np.vstack([self.A, other.A]), self.framework, self.device),
                iop.to_array(np.concatenate([self.b, other.b]), self.framework, self.device),
            )
        return SumCost([self, other])
