from __future__ import annotations

from functools import cached_property

import numpy as np
import numpy.linalg as la
from numpy import float64
from numpy.typing import NDArray
from scipy import special

import decent_bench.centralized_algorithms as ca
import decent_bench.utils.interoperability as iop
from decent_bench.costs._base._cost import Cost
from decent_bench.costs._base._sum_cost import SumCost
from decent_bench.utils.array import Array
from decent_bench.utils.types import (
    Dataset,
    EmpiricalRiskBatchSize,
    EmpiricalRiskIndices,
    EmpiricalRiskReduction,
    SupportedDevices,
    SupportedFrameworks,
)

from ._empirical_risk_cost import EmpiricalRiskCost


class LogisticRegressionCost(EmpiricalRiskCost):
    r"""
    Logistic regression cost function.

    Given a data matrix :math:`\mathbf{A} \in \mathbb{R}^{m \times n}` and target vector
    :math:`\mathbf{b} \in \mathbb{R}^{m}`, the logistic regression cost function is defined as:

    .. math:: f(\mathbf{x}) =
        -\frac{1}{m} \left[ \mathbf{b}^T \log( \sigma(\mathbf{Ax}) )
        + ( \mathbf{1} - \mathbf{b} )^T
            \log( 1 - \sigma(\mathbf{Ax}) ) \right]

        = -\frac{1}{m} \sum_{i = 1}^m \left[ b_i \log( \sigma(A_i x) )
        + (1 - b_i) \log( 1 - \sigma(A_i x) ) \right]

    where :math:`A_i` and :math:`b_i` are the i-th row of :math:`\mathbf{A}` and
    the i-th element of :math:`\mathbf{b}` respectively.

    In the stochastic setting, a mini-batch of size :math:`b < m` is used to compute the cost and its derivatives.
    The cost function then becomes:

    .. math:: f(\mathbf{x}) =
        -\frac{1}{b} \left[ \mathbf{b}_{\mathcal{B}}^T \log( \sigma(\mathbf{A}_{\mathcal{B}}\mathbf{x}) )
        + ( \mathbf{1} - \mathbf{b}_{\mathcal{B}} )^T
            \log( 1 - \sigma(\mathbf{A}_{\mathcal{B}}\mathbf{x}) ) \right]

        = -\frac{1}{b} \sum_{i \in \mathcal{B}} \left[ b_i \log( \sigma(A_i x) )
        + (1 - b_i) \log( 1 - \sigma(A_i x) ) \right]

    where :math:`\mathcal{B}` is a sampled batch of :math:`b` indices from :math:`\{1, \ldots, m\}`,
    :math:`\mathbf{A}_B` and :math:`\mathbf{b}_B` are the rows corresponding to the batch :math:`\mathcal{B}`.
    """

    def __init__(self, dataset: Dataset, batch_size: EmpiricalRiskBatchSize = "all"):
        """
        Initialize logistic regression cost function.

        Args:
            dataset (Dataset): Dataset containing features and targets. The expected shapes are:
                - Features: (n_features,)
                - Targets: single dimensional values
            batch_size (EmpiricalRiskBatchSize): Size of mini-batch to use for stochastic methods.
                If "all", full-batch methods are used.

        Raises:
            ValueError: If input dimensions are incorrect or batch_size is invalid.
            TypeError: If dataset targets are not single dimensional values.

        """
        if len(iop.shape(dataset[0][0])) != 1:
            raise ValueError(f"Dataset features must be vectors, got: {dataset[0][0]}")
        if iop.to_numpy(dataset[0][1]).shape != (1,):
            raise TypeError(
                f"Dataset targets must be single dimensional values, got: {dataset[0][1]} "
                f"with shape {iop.to_numpy(dataset[0][1]).shape}, expected shape is (1,)."
            )
        if isinstance(batch_size, int) and (batch_size <= 0 or batch_size > len(dataset)):
            raise ValueError(
                f"Batch size must be positive and at most the number of samples, "
                f"got: {batch_size} and number of samples is: {len(dataset)}."
            )
        if isinstance(batch_size, str) and batch_size != "all":
            raise ValueError(f"Invalid batch size string. Supported value is 'all', got {batch_size}.")

        class_labels = {iop.to_numpy(y).item() for _, y in dataset}
        if len(class_labels) != 2:
            raise ValueError("Dataset must contain exactly two classes")

        self._dataset = dataset
        self._label_mapping = dict(enumerate(class_labels))
        self._batch_size = self.n_samples if batch_size == "all" else batch_size
        # Cache data matrices for efficiency when using full dataset
        self.A: NDArray[float64] | None = None
        self.b: NDArray[float64] | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        return iop.shape(self._dataset[0][0])

    @property
    def framework(self) -> SupportedFrameworks:
        return SupportedFrameworks.NUMPY

    @property
    def device(self) -> SupportedDevices:
        return SupportedDevices.CPU

    @property
    def n_samples(self) -> int:
        return len(self._dataset)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    @cached_property
    def m_smooth(self) -> float:  # pyright: ignore[reportIncompatibleMethodOverride]
        r"""
        The cost function's smoothness constant.

        .. math::
            \frac{1}{m} \frac{m}{4} \max_i \|\mathbf{A}_i\|^2 = \frac{1}{4} \max_i \|\mathbf{A}_i\|^2

        where m is the number of rows in :math:`\mathbf{A}`.

        For the general definition, see
        :attr:`Cost.m_smooth <decent_bench.costs.Cost.m_smooth>`.
        """
        A, _ = self._get_batch_data("all")  # noqa: N806
        return float(max(pow(la.norm(row), 2) for row in A) / 4)

    @property
    def m_cvx(self) -> float:
        """
        The cost function's convexity constant, 0.

        For the general definition, see
        :attr:`Cost.m_cvx <decent_bench.costs.Cost.m_cvx>`.
        """
        return 0

    @iop.autodecorate_cost_method(EmpiricalRiskCost.predict)
    def predict(self, x: NDArray[float64], data: list[NDArray[float64]]) -> NDArray[float64]:
        r"""
        Make predictions at x on the given data.

        The predicted targets are computed as :math:`\sigma(\mathbf{Ax}) > 0.5`.

        Args:
            x: Point to make predictions at.
            data: List of NDArray containing data to make predictions on.

        Returns:
            Predicted targets as an array.

        """
        pred_data = np.stack(data) if isinstance(data, list) else data
        logits = pred_data.dot(x)
        sig = special.expit(logits)
        return np.array([self._label_mapping[label] for label in (sig >= 0.5).astype(int)])

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
            - \frac{1}{m} \left[ \mathbf{b}^T \log( \sigma(\mathbf{Ax}) )
            + ( \mathbf{1} - \mathbf{b} )^T
                \log( 1 - \sigma(\mathbf{Ax}) ) \right]

        If indices is "batch", a random batch :math:`\mathcal{B}` is drawn with :attr:`batch_size` samples.

        .. math::
            -\frac{1}{b} \left[ \mathbf{b}_{\mathcal{B}}^T \log( \sigma(\mathbf{A}_{\mathcal{B}}\mathbf{x}) )
            + ( \mathbf{1} - \mathbf{b}_{\mathcal{B}} )^T
                \log( 1 - \sigma(\mathbf{A}_{\mathcal{B}}\mathbf{x}) ) \right]

        where :math:`\mathbf{A}_B` and :math:`\mathbf{b}_B` are the rows corresponding to the batch :math:`\mathcal{B}`.

        """
        A, b = self._get_batch_data(indices)  # noqa: N806
        Ax = A.dot(x)  # noqa: N806
        neg_log_sig = np.logaddexp(0.0, -Ax)
        cost = b.dot(neg_log_sig) + (1 - b).dot(Ax + neg_log_sig)
        return float(cost) / len(self.batch_used)

    @iop.autodecorate_cost_method(EmpiricalRiskCost.gradient)
    def gradient(
        self,
        x: NDArray[float64],
        indices: EmpiricalRiskIndices = "batch",
        reduction: EmpiricalRiskReduction = "mean",
    ) -> NDArray[float64]:
        r"""
        Gradient at x using datapoints at the given indices.

        Supported values for indices are:
            - int: the corresponding datapoint is used.
            - list[int]: corresponding datapoints are used.
            - "all": the full dataset is used.
            - "batch": a batch is drawn with :attr:`batch_size` samples.

        Supported values for reduction are:
            - "mean": average the gradients over the samples.
            - None: return the gradients for each sample, index as the first dimension.

        If no batching is used, this is:

        .. math::
            \frac{1}{m} \mathbf{A}^T (\sigma(\mathbf{Ax}) - \mathbf{b})

        If indices is "batch", a random batch :math:`\mathcal{B}` is drawn with :attr:`batch_size` samples.

        .. math::
            \frac{1}{b} \mathbf{A}_{\mathcal{B}}^T (\sigma(\mathbf{A}_{\mathcal{B}}\mathbf{x})
            - \mathbf{b}_{\mathcal{B}})

        where :math:`\mathbf{A}_B` and :math:`\mathbf{b}_B` are the rows corresponding to the batch :math:`\mathcal{B}`.

        Note:
            When reduction is None, the returned array will have an additional leading dimension
            corresponding to the number of samples used. Indexing into this dimension will give the gradient
            for the respective sample in :attr:`batch_used <decent_bench.costs.EmpiricalRiskCost.batch_used>`.

        """
        if reduction is None:
            return self._per_sample_gradients(x, indices)

        A, b = self._get_batch_data(indices)  # noqa: N806
        sig = special.expit(A.dot(x))
        res: NDArray[float64] = A.T.dot(sig - b) / len(self.batch_used)
        return res

    def _per_sample_gradients(
        self,
        x: NDArray[float64],
        indices: EmpiricalRiskIndices = "batch",
    ) -> NDArray[float64]:
        A, b = self._get_batch_data(indices)  # noqa: N806
        sig = special.expit(A.dot(x))
        res = [A[i, :].reshape(-1, 1) * (sig[i] - b[i]) for i in range(A.shape[0])]
        return np.asarray(res)

    @iop.autodecorate_cost_method(EmpiricalRiskCost.hessian)
    def hessian(self, x: NDArray[float64], indices: EmpiricalRiskIndices = "batch") -> NDArray[float64]:
        r"""
        Hessian at x using datapoints at the given indices.

        Supported values for indices are:
            - int: the corresponding datapoint is used.
            - list[int]: corresponding datapoints are used.
            - "all": the full dataset is used.
            - "batch": a batch is drawn with :attr:`batch_size` samples.

        If no batching is used, this is:

        .. math::
            \frac{1}{m} \mathbf{A}^T \mathbf{DA}

        where :math:`\mathbf{D}` is a diagonal matrix such that
        :math:`\mathbf{D}_i = \sigma(\mathbf{Ax}_i) (1-\sigma(\mathbf{Ax}_i))`

        If indices is "batch", a random batch :math:`\mathcal{B}` is drawn with :attr:`batch_size` samples.

        .. math::
            \frac{1}{b} \mathbf{A}_{\mathcal{B}}^T \mathbf{D}_{\mathcal{B}} \mathbf{A}_{\mathcal{B}}

        where :math:`\mathbf{A}_B` and :math:`\mathbf{D}_B` are the rows corresponding to the batch :math:`\mathcal{B}`.
        """
        A, _ = self._get_batch_data(indices)  # noqa: N806
        sig = special.expit(A.dot(x))
        D = np.diag(sig * (1 - sig))  # noqa: N806
        res: NDArray[float64] = A.T.dot(D).dot(A) / len(self.batch_used)
        return res

    @iop.autodecorate_cost_method(EmpiricalRiskCost.proximal)
    def proximal(self, x: Array, rho: float) -> Array:
        """
        Proximal at x solved using an iterative method.

        The proximal for logistic regression does not have closed form solution, will use
        a gradient based approximation method over the entire dataset.

        See
        :meth:`Cost.proximal() <decent_bench.costs.Cost.proximal>`
        for the general proximal definition.

        """
        prev_batch_size = self.batch_size
        self._batch_size = self.n_samples  # Use full dataset for proximal
        approx = ca.proximal_solver(self, x, rho)
        self._batch_size = prev_batch_size  # Restore previous batch size
        return approx / self.n_samples

    def _get_batch_data(self, indices: EmpiricalRiskIndices = "batch") -> tuple[NDArray[float64], NDArray[float64]]:
        """Get data for a batch. Returns A and b for the batch."""
        indices = self._sample_batch_indices(indices)

        if len(indices) == self.n_samples:
            # Use full dataset
            if self.A is None or self.b is None:
                self.A = np.stack([iop.to_numpy(x) for x, _ in self._dataset])
                self.b = np.stack([iop.to_numpy(y) for _, y in self._dataset]).squeeze()
                for k in self._label_mapping:
                    self.b[np.where(self.b == self._label_mapping[k])] = k
            return self.A, self.b

        A_list, b_list = [], []  # noqa: N806
        for idx in indices:
            x_i, y_i = self._dataset[idx]
            A_list.append(iop.to_numpy(x_i))
            b_list.append(iop.to_numpy(y_i))
        A = np.stack(A_list)  # noqa: N806
        b = np.stack(b_list).squeeze()
        for k in self._label_mapping:
            b[np.where(b == self._label_mapping[k])] = k
        return A, b

    def __add__(self, other: Cost) -> Cost:
        """
        Add another cost function.

        Raises:
            ValueError: if the domain shapes don't match

        """
        if self.shape != other.shape:
            raise ValueError(f"Mismatching domain shapes: {self.shape} vs {other.shape}")
        if isinstance(other, LogisticRegressionCost):
            if self.batch_size == self.n_samples and other.batch_size == other.n_samples:
                combined_batch_size = self.n_samples + other.n_samples
            elif self.batch_size == self.n_samples:
                combined_batch_size = other.batch_size
            elif other.batch_size == other.n_samples:
                combined_batch_size = self.batch_size
            else:
                combined_batch_size = max(self.batch_size, other.batch_size)

            return LogisticRegressionCost(
                dataset=self._dataset + other._dataset,
                batch_size=combined_batch_size,
            )
        return SumCost([self, other])
