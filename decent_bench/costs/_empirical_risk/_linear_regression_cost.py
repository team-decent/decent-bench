from __future__ import annotations

from functools import cached_property

import numpy as np
from numpy import float64
from numpy.typing import NDArray

import decent_bench.utils.interoperability as iop
from decent_bench.costs._base._cost import Cost
from decent_bench.costs._base._sum_cost import SumCost
from decent_bench.costs._empirical_risk._empirical_risk_cost import EmpiricalRiskCost
from decent_bench.utils.array import Array
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks


class LinearRegressionCost(EmpiricalRiskCost):
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

    def __init__(self, A: Array, b: Array, batch_size: int | None = None):  # noqa: N803
        if len(iop.shape(A)) != 2:
            raise ValueError("Matrix A must be 2D")
        if len(iop.shape(b)) != 1:
            raise ValueError("Vector b must be 1D")
        if iop.shape(A)[0] != iop.shape(b)[0]:
            raise ValueError(f"Dimension mismatch: A has {iop.shape(A)[0]} rows but b has {iop.shape(b)[0]} elements")

        self.A: NDArray[float64] = iop.to_numpy(A)
        self.b: NDArray[float64] = iop.to_numpy(iop.copy(b))  # Copy b to avoid modifying original array pointer
        class_labels = np.unique(self.b)
        self._label_mapping = {i: class_labels[i] for i in range(len(class_labels))}
        for i, label in self._label_mapping.items():
            self.b[np.where(self.b == label)] = i
        self.ATA: NDArray[float64] = self.A.T @ self.A
        self._batch_size = batch_size

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
        return int(self.A.shape[0])

    @property
    def batch_size(self) -> int | None:
        return self._batch_size

    @cached_property
    def m_smooth(self) -> float:  # pyright: ignore[reportIncompatibleMethodOverride]
        r"""
        The cost function's smoothness constant.

        .. math::
            \max_{i} \left| \lambda_i \right|

        where :math:`\lambda_i` are the eigenvalues of :math:`\mathbf{A}^T \mathbf{A}`.

        For the general definition, see
        :attr:`Cost.m_smooth <decent_bench.costs.Cost.m_smooth>`.
        """
        eigs = np.linalg.eigvalsh(self.ATA)
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

        where :math:`\lambda_i` are the eigenvalues of :math:`\mathbf{A}^T \mathbf{A}`.

        For the general definition, see
        :attr:`Cost.m_cvx <decent_bench.costs.Cost.m_cvx>`.
        """
        l_min = float(np.min(np.linalg.eigvalsh(self.ATA)))
        tol = 1e-12
        if l_min > tol:
            return l_min
        if abs(l_min) <= tol:
            return 0
        return np.nan

    @iop.autodecorate_cost_method(EmpiricalRiskCost.predict)
    def predict(self, x: NDArray[float64], data: list[NDArray[float64]]) -> NDArray[float64]:
        """
        Make predictions at x on the given data.

        Args:
            x: Point to make predictions at.
            data: List of NDArray containing data to make predictions on.

        Returns:
            Predicted labels as an array.

        """
        pred_data = np.stack(data) if isinstance(data, list) else data
        continuous_predictions = pred_data.dot(x)
        return np.array([self._label_mapping[label] for label in (continuous_predictions >= 0.5).astype(int)])

    @iop.autodecorate_cost_method(EmpiricalRiskCost.function)
    def function(self, x: NDArray[float64], indices: list[int] | None = None) -> float:
        r"""
        Evaluate function at x using datapoints at the given indices.

        If indices is None, a random batch is drawn with :attr:`batch_size` samples.

        .. math:: \frac{1}{2} \| \mathbf{Ax} - \mathbf{b} \|^2
        """
        if indices is None:
            indices = self._sample_batch_indices()
        A, _, b = self._get_batch_data(indices)  # noqa: N806
        residual = A.dot(x) - b
        return float(0.5 * residual.dot(residual))

    @iop.autodecorate_cost_method(EmpiricalRiskCost.gradient)
    def gradient(self, x: NDArray[float64], indices: list[int] | None = None) -> NDArray[float64]:
        r"""
        Gradient at x using datapoints at the given indices.

        If indices is None, a random batch is drawn with :attr:`batch_size` samples.

        .. math:: \mathbf{A}^T\mathbf{Ax} - \mathbf{A}^T \mathbf{b}
        """
        if indices is None:
            indices = self._sample_batch_indices()
        A, ATA, b = self._get_batch_data(indices)  # noqa: N806
        res: NDArray[float64] = ATA.dot(x) - A.T.dot(b)
        return res

    @iop.autodecorate_cost_method(EmpiricalRiskCost.hessian)
    def hessian(self, x: NDArray[float64], indices: list[int] | None = None) -> NDArray[float64]:  # noqa: ARG002
        r"""
        Hessian at x using datapoints at the given indices.

        If indices is None, a random batch is drawn with :attr:`batch_size` samples.

        .. math:: \mathbf{A}^T\mathbf{A}
        """
        if indices is None:
            indices = self._sample_batch_indices()
        _, ATA, _ = self._get_batch_data(indices)  # noqa: N806
        res: NDArray[float64] = ATA
        return res

    @iop.autodecorate_cost_method(EmpiricalRiskCost.proximal)
    def proximal(self, x: NDArray[float64], rho: float, indices: list[int] | None = None) -> NDArray[float64]:
        r"""
        Proximal at x using datapoints at the given indices.

        If indices is None, a random batch is drawn with :attr:`batch_size` samples.

        .. math::
            (\rho \mathbf{A}^T \mathbf{A} + \mathbf{I})^{-1} (\mathbf{x} + \rho \mathbf{A}^T\mathbf{b})

        where :math:`\rho > 0` is the penalty.

        This is a closed form solution, see
        :meth:`Cost.proximal() <decent_bench.costs.Cost.proximal>`
        for the general proximal definition.
        """
        if indices is None:
            indices = self._sample_batch_indices()
        A, ATA, b = self._get_batch_data(indices)  # noqa: N806
        lhs = rho * ATA + np.eye(A.shape[1])
        rhs = x + rho * A.T @ b
        return np.asarray(np.linalg.solve(lhs, rhs), dtype=float64)

    def _get_batch_data(self, indices: list[int]) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]:
        """Get data for a batch. Returns A, A.T@A and b for the batch."""
        if len(indices) == self.n_samples:
            return self.A, self.ATA, self.b
        A, b = self.A[indices, :], self.b[indices]  # noqa: N806
        return A, A.T @ A, b

    def __add__(self, other: Cost) -> Cost:
        """
        Add another cost function.

        Raises:
            ValueError: if the domain shapes don't match

        """
        if self.shape != other.shape:
            raise ValueError(f"Mismatching domain shapes: {self.shape} vs {other.shape}")
        if isinstance(other, LinearRegressionCost):
            return LinearRegressionCost(
                iop.to_array(np.vstack([self.A, other.A]), self.framework, self.device),
                iop.to_array(np.hstack([self.b, other.b]), self.framework, self.device),
                batch_size=int(np.mean([self.batch_size, other.batch_size]))
                if self.batch_size and other.batch_size
                else None,
            )
        return SumCost([self, other])
