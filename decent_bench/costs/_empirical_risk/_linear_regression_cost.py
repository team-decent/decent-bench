from __future__ import annotations

from functools import cached_property
from typing import Literal

import numpy as np
from numpy import float64
from numpy.typing import NDArray

import decent_bench.utils.interoperability as iop
from decent_bench.costs._base._cost import Cost
from decent_bench.costs._base._sum_cost import SumCost
from decent_bench.costs._empirical_risk._empirical_risk_cost import EmpiricalRiskCost
from decent_bench.utils.array import Array
from decent_bench.utils.types import EmpiricalRiskIndices, SupportedDevices, SupportedFrameworks


class LinearRegressionCost(EmpiricalRiskCost):
    r"""
    Linear regression cost function.

    Given a data matrix :math:`\mathbf{A} \in \mathbb{R}^{m \times n}` and target vector
    :math:`\mathbf{b} \in \mathbb{R}^{m}`, the linear regression cost function is defined as:

    .. math::
        f(\mathbf{x}) = \frac{1}{2} \| \mathbf{Ax} - \mathbf{b} \|^2

        = \frac{1}{2} \sum_{i = 1}^m (A_i x - b_i)^2

    where :math:`A_i` and :math:`b_i` are the i-th row of :math:`\mathbf{A}` and
    the i-th element of :math:`\mathbf{b}` respectively.

    In the stochastic setting, a mini-batch of size :math:`b < m` is used to compute the cost and its derivatives.
    The cost function then becomes:

    .. math::
        f(\mathbf{x}) = \frac{1}{2} \| \mathbf{A}_{\mathcal{B}}\mathbf{x} - \mathbf{b}_{\mathcal{B}} \|^2

        = \frac{1}{2} \sum_{i \in \mathcal{B}} (A_i x - b_i)^2

    where :math:`\mathcal{B}` is a sampled batch of :math:`b` indices from :math:`\{1, \ldots, m\}`,
    :math:`\mathbf{A}_B` and :math:`\mathbf{b}_B` are the rows corresponding to the batch :math:`\mathcal{B}`.
    """

    def __init__(self, A: Array, b: Array, batch_size: int | Literal["all"] = "all"):  # noqa: N803
        """
        Initialize a LinearRegressionCost instance.

        Args:
            A (Array): Data matrix of shape (n_samples, shape).
            b (Array): Target vector of shape (n_samples,).
            batch_size (int | Literal["all"]): Size of mini-batches for stochastic methods, or "all" for full-batch.

        Raises:
            ValueError: If input dimensions are inconsistent or batch_size is invalid.

        """
        if len(iop.shape(A)) != 2:
            raise ValueError("Matrix A must be 2D")
        if len(iop.shape(b)) != 1:
            raise ValueError("Vector b must be 1D")
        if iop.shape(A)[0] != iop.shape(b)[0]:
            raise ValueError(f"Dimension mismatch: A has {iop.shape(A)[0]} rows but b has {iop.shape(b)[0]} elements")
        if isinstance(batch_size, int) and (batch_size <= 0 or batch_size > iop.shape(A)[0]):
            raise ValueError(
                f"Batch size must be positive and at most the number of samples, "
                f"got: {batch_size} and number of samples is: {iop.shape(A)[0]}."
            )
        if isinstance(batch_size, str) and batch_size != "all":
            raise ValueError(f"Invalid batch size string. Supported value is 'all', got {batch_size}.")

        self.A: NDArray[float64] = iop.to_numpy(A)
        self.b: NDArray[float64] = iop.to_numpy(b)
        self.ATA: NDArray[float64] = self.A.T @ self.A
        self._batch_size = self.n_samples if batch_size == "all" else batch_size

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
    def batch_size(self) -> int:
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
        r"""
        Make predictions at x on the given data.

        The predicted targets are computed as :math:`\mathbf{Ax}`.

        Args:
            x: Point to make predictions at.
            data: List of NDArray containing data to make predictions on.

        Returns:
            Predicted targets as an array.

        """
        pred_data = np.stack(data) if isinstance(data, list) else data
        pred: NDArray[float64] = pred_data.dot(x)
        return pred

    @iop.autodecorate_cost_method(EmpiricalRiskCost.function)
    def function(self, x: NDArray[float64], indices: EmpiricalRiskIndices = "batch") -> float:
        r"""
        Evaluate function at x using datapoints at the given indices.

        Supported values for indices are:
            - int: the corresponding datapoint is used.
            - list[int]: corresponding datapoints are used.
            - "all": the full dataset is used.
            - "batch": a batch is drawn with :attr:`batch_size` samples.

        If no batching is used, this is:

        .. math::
            \frac{1}{2} \| \mathbf{Ax} - \mathbf{b} \|^2

        If indices is "batch", a random batch :math:`\mathcal{B}` is drawn with :attr:`batch_size` samples.

        .. math::
            \frac{1}{2} \| \mathbf{A}_{\mathcal{B}}\mathbf{x} - \mathbf{b}_{\mathcal{B}} \|^2

        where :math:`\mathbf{A}_B` and :math:`\mathbf{b}_B` are the rows corresponding to the batch :math:`\mathcal{B}`.
        """
        A, _, b = self._get_batch_data(indices)  # noqa: N806
        residual = A.dot(x) - b
        return float(0.5 * residual.dot(residual))

    @iop.autodecorate_cost_method(EmpiricalRiskCost.gradient)
    def gradient(self, x: NDArray[float64], indices: EmpiricalRiskIndices = "batch") -> NDArray[float64]:
        r"""
        Gradient at x using datapoints at the given indices.

        Supported values for indices are:
            - int: the corresponding datapoint is used.
            - list[int]: corresponding datapoints are used.
            - "all": the full dataset is used.
            - "batch": a batch is drawn with :attr:`batch_size` samples.

        If no batching is used, this is:

        .. math::
            \mathbf{A}^T\mathbf{Ax} - \mathbf{A}^T \mathbf{b}

        If indices is "batch", a random batch :math:`\mathcal{B}` is drawn with :attr:`batch_size` samples.

        .. math::
            \mathbf{A}_{\mathcal{B}}^T\mathbf{A}_{\mathcal{B}}\mathbf{x} -
            \mathbf{A}_{\mathcal{B}}^T \mathbf{b}_{\mathcal{B}}

        where :math:`\mathbf{A}_B` and :math:`\mathbf{b}_B` are the rows corresponding to the batch :math:`\mathcal{B}`.

        """
        A, ATA, b = self._get_batch_data(indices)  # noqa: N806
        res: NDArray[float64] = ATA.dot(x) - A.T.dot(b)
        return res

    @iop.autodecorate_cost_method(EmpiricalRiskCost.hessian)
    def hessian(self, x: NDArray[float64], indices: EmpiricalRiskIndices = "batch") -> NDArray[float64]:  # noqa: ARG002
        r"""
        Hessian at x using datapoints at the given indices.

        Supported values for indices are:
            - int: the corresponding datapoint is used.
            - list[int]: corresponding datapoints are used.
            - "all": the full dataset is used.
            - "batch": a batch is drawn with :attr:`batch_size` samples.

        If no batching is used, this is:

        .. math::
            \mathbf{A}^T\mathbf{A}

        If indices is "batch", a random batch :math:`\mathcal{B}` is drawn with :attr:`batch_size` samples.

        .. math::
            \mathbf{A}_{\mathcal{B}}^T \mathbf{A}_{\mathcal{B}}

        where :math:`\mathbf{A}_B` and :math:`\mathbf{b}_B` are the rows corresponding to the batch :math:`\mathcal{B}`.
        """
        _, ATA, _ = self._get_batch_data(indices)  # noqa: N806
        res: NDArray[float64] = ATA
        return res

    @iop.autodecorate_cost_method(EmpiricalRiskCost.proximal)
    def proximal(self, x: NDArray[float64], rho: float) -> NDArray[float64]:
        r"""
        Proximal at x using the full dataset.

        The proximal operator for the linear regression cost function is given by:

        .. math::
            (\rho \mathbf{A}^T \mathbf{A} + \mathbf{I})^{-1} (\mathbf{x} + \rho \mathbf{A}^T\mathbf{b})

        where :math:`\rho > 0` is the penalty. This is a closed form solution.

        """
        A, ATA, b = self._get_batch_data("all")  # noqa: N806
        lhs = rho * ATA + np.eye(A.shape[1])
        rhs = x + rho * A.T @ b
        return np.asarray(np.linalg.solve(lhs, rhs), dtype=float64)

    def _get_batch_data(
        self,
        indices: EmpiricalRiskIndices = "batch",
    ) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]:
        """Get data for a batch. Returns A, A.T@A and b for the batch."""
        indices = self._sample_batch_indices(indices)

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
            if self.batch_size == self.n_samples and other.batch_size == other.n_samples:
                combined_batch_size = self.n_samples + other.n_samples
            elif self.batch_size == self.n_samples:
                combined_batch_size = other.batch_size
            elif other.batch_size == other.n_samples:
                combined_batch_size = self.batch_size
            else:
                combined_batch_size = max(self.batch_size, other.batch_size)

            return LinearRegressionCost(
                iop.to_array(np.vstack([self.A, other.A]), self.framework, self.device),
                iop.to_array(np.hstack([self.b, other.b]), self.framework, self.device),
                batch_size=combined_batch_size,
            )
        return SumCost([self, other])
