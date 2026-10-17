from __future__ import annotations

from functools import cached_property
from typing import Any

from decent_array import Array
from decent_array import interoperability as iop

import decent_bench.utils.solvers as ca
from decent_bench.utils._tags import Tag, tags
from decent_bench.utils.types import (
    Dataset,
    EmpiricalRiskBatchSize,
    EmpiricalRiskIndices,
    EmpiricalRiskReduction,
)

from ._empirical_risk_cost import EmpiricalRiskCost


def _sigmoid(x: Array) -> Array:
    return iop.exp(-iop.logaddexp(iop.zeros_like(x), -x))


@tags(Tag.COST, Tag.CLASSIFICATION, Tag.EMPIRICAL_RISK)
class LogisticRegressionCost(EmpiricalRiskCost):
    r"""
    Logistic regression cost function.

    Given a data matrix :math:`\mathbf{A} \in \mathbb{R}^{m \times n}` and target vector
    :math:`\mathbf{b} \in \mathbb{R}^{m}`, the logistic regression cost function is defined as:

    .. math:: f(\mathbf{x}) =
        -\frac{1}{m}\left[ \mathbf{b}^T \log( \sigma(\mathbf{Ax}) )
        + ( \mathbf{1} - \mathbf{b} )^T
            \log( 1 - \sigma(\mathbf{Ax}) ) \right]

        = -\frac{1}{m}\sum_{i = 1}^m \left[ b_i \log( \sigma(A_i x) )
        + (1 - b_i) \log( 1 - \sigma(A_i x) ) \right]

    where :math:`\sigma(z) = \frac{1}{1 + e^{-z}}` is the sigmoid function, :math:`A_i` and :math:`b_i` are the i-th
    row of :math:`\mathbf{A}` and the i-th element of :math:`\mathbf{b}` respectively.

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
        if dataset[0][0].ndim != 1:
            raise ValueError(f"Dataset features must be vectors, got: {dataset[0][0].shape}")
        if dataset[0][1].shape != (1,):
            raise TypeError(f"Dataset targets must be single dimensional values, got: {dataset[0][1].shape}")
        if isinstance(batch_size, int) and (batch_size <= 0 or batch_size > len(dataset)):
            raise ValueError(
                f"Batch size must be positive and at most the number of samples, "
                f"got: {batch_size} and number of samples is: {len(dataset)}."
            )
        if isinstance(batch_size, str) and batch_size != "all":
            raise ValueError(f"Invalid batch size string. Supported value is 'all', got {batch_size}.")

        class_labels = {iop.squeeze(y).item() for _, y in dataset}
        if len(class_labels) != 2:
            raise ValueError("Dataset must contain exactly two classes")

        self._dataset = dataset
        self._label_mapping = dict(enumerate(class_labels))
        self._batch_size = self.n_samples if batch_size == "all" else batch_size
        # Cache data matrices for efficiency when using full dataset
        self.A: Array | None = None
        self.b: Array | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        return iop.shape(self._dataset[0][0])

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
    def m_smooth(self) -> float:
        r"""
        The cost function's smoothness constant.

        .. math::
            \frac{1}{m} \frac{m}{4} \max_i \|\mathbf{A}_i\|^2 = \frac{1}{4} \max_i \|\mathbf{A}_i\|^2

        where m is the number of rows in :math:`\mathbf{A}`.

        For the general definition, see
        :attr:`Cost.m_smooth <decent_bench.costs.Cost.m_smooth>`.
        """
        A, _ = self._get_batch_data("all")  # noqa: N806
        return max(pow(float(iop.norm(A[i, :])), 2) for i in range(A.shape[0])) / 4

    @property
    def m_cvx(self) -> float:
        """
        The cost function's convexity constant, 0.

        For the general definition, see
        :attr:`Cost.m_cvx <decent_bench.costs.Cost.m_cvx>`.
        """
        return 0

    def predict(self, x: Array, data: list[Array]) -> Array:
        r"""
        Make predictions at x on the given data.

        The predicted targets are computed as :math:`\sigma(\mathbf{Ax}) > 0.5`,
        where :math:`\sigma` is the sigmoid function.

        Args:
            x: Point to make predictions at.
            data: List of NDArray containing data to make predictions on.

        Returns:
            Predicted targets as an array.

        """
        logits = iop.stack(data) @ x
        sig = _sigmoid(logits)
        return iop.where(sig >= 0.5, self._label_mapping[1], self._label_mapping[0])

    def function(self, x: Array, indices: EmpiricalRiskIndices = "batch", **kwargs: Any) -> float:  # noqa: ARG002, ANN401
        r"""
        Evaluate function at x using datapoints at the given indices.

        Supported values for indices are:
            - int: datapoint to use.
            - list[int]: datapoints to use.
            - "all": use the full dataset.
            - "batch": draw a batch with :attr:`batch_size` samples.

        If no batching is used, this is:

        .. math::
            -\frac{1}{m}\left[ \mathbf{b}^T \log( \sigma(\mathbf{Ax}) )
            + ( \mathbf{1} - \mathbf{b} )^T
                \log( 1 - \sigma(\mathbf{Ax}) ) \right]

        If indices is "batch", a random batch :math:`\mathcal{B}` is drawn with :attr:`batch_size` samples.

        .. math::
            -\frac{1}{b} \left[ \mathbf{b}_{\mathcal{B}}^T \log( \sigma(\mathbf{A}_{\mathcal{B}}\mathbf{x}) )
            + ( \mathbf{1} - \mathbf{b}_{\mathcal{B}} )^T
                \log( 1 - \sigma(\mathbf{A}_{\mathcal{B}}\mathbf{x}) ) \right]

        where :math:`\sigma` is the sigmoid function, :math:`\mathbf{A}_B` and :math:`\mathbf{b}_B` are
        the rows corresponding to the batch :math:`\mathcal{B}`.

        """
        A, b = self._get_batch_data(indices)  # noqa: N806
        Ax = iop.matmul(A, x)  # noqa: N806
        neg_log_sig = iop.logaddexp(iop.zeros_like(Ax), -Ax)
        cost = iop.sum(b * neg_log_sig + (1.0 - b) * (Ax + neg_log_sig))
        return float(cost.item()) / len(self.batch_used)

    def gradient(
        self,
        x: Array,
        indices: EmpiricalRiskIndices = "batch",
        reduction: EmpiricalRiskReduction = "mean",
        **kwargs: Any,  # noqa: ARG002, ANN401
    ) -> Array:
        r"""
        Gradient at x using datapoints at the given indices.

        Supported values for indices are:
            - int: datapoint to use.
            - list[int]: datapoints to use.
            - "all": use the full dataset.
            - "batch": draw a batch with :attr:`batch_size` samples.

        Supported values for reduction are:
            - "mean": average the gradients over the samples.
            - None: return the gradients for each sample, index as the first dimension.

        If no batching is used, this is:

        .. math::
            \frac{1}{m}\mathbf{A}^T (\sigma(\mathbf{Ax}) - \mathbf{b})

        If indices is "batch", a random batch :math:`\mathcal{B}` is drawn with :attr:`batch_size` samples.

        .. math::
            \frac{1}{b} \mathbf{A}_{\mathcal{B}}^T (\sigma(\mathbf{A}_{\mathcal{B}}\mathbf{x})
            - \mathbf{b}_{\mathcal{B}})

        where :math:`\sigma` is the sigmoid function, :math:`\mathbf{A}_B` and :math:`\mathbf{b}_B`
        are the rows corresponding to the batch :math:`\mathcal{B}`.

        Note:
            When reduction is None, the returned array will have an additional leading dimension
            corresponding to the number of samples used. Indexing into this dimension will give the gradient
            for the respective sample in :attr:`batch_used <decent_bench.costs.EmpiricalRiskCost.batch_used>`.

        """
        if reduction is None:
            return self._per_sample_gradients(x, indices)

        A, b = self._get_batch_data(indices)  # noqa: N806
        sig = _sigmoid(A @ x)
        return (A.T @ (sig - b)) / len(self.batch_used)

    def _per_sample_gradients(
        self,
        x: Array,
        indices: EmpiricalRiskIndices = "batch",
    ) -> Array:
        A, b = self._get_batch_data(indices)  # noqa: N806
        sig = _sigmoid(A @ x)
        residuals = sig - b
        return iop.expand_dims(residuals, axis=1) * A

    def hessian(self, x: Array, indices: EmpiricalRiskIndices = "batch", **kwargs: Any) -> Array:  # noqa: ARG002, ANN401
        r"""
        Hessian at x using datapoints at the given indices.

        Supported values for indices are:
            - int: datapoint to use.
            - list[int]: datapoints to use.
            - "all": use the full dataset.
            - "batch": draw a batch with :attr:`batch_size` samples.

        If no batching is used, this is:

        .. math::
            \frac{1}{m}\mathbf{A}^T \mathbf{DA}

        where :math:`\sigma` is the sigmoid function and :math:`\mathbf{D}` is a diagonal matrix such that
        :math:`\mathbf{D}_i = \sigma(\mathbf{Ax}_i) (1-\sigma(\mathbf{Ax}_i))`

        If indices is "batch", a random batch :math:`\mathcal{B}` is drawn with :attr:`batch_size` samples.

        .. math::
            \frac{1}{b} \mathbf{A}_{\mathcal{B}}^T \mathbf{D}_{\mathcal{B}} \mathbf{A}_{\mathcal{B}}

        where :math:`\mathbf{A}_B` and :math:`\mathbf{D}_B` are the rows corresponding to the batch :math:`\mathcal{B}`.
        """
        A, _ = self._get_batch_data(indices)  # noqa: N806
        sig = _sigmoid(A @ x)
        weights = sig * (1.0 - sig)
        weighted_A = iop.expand_dims(weights, axis=1) * A  # noqa: N806
        return (A.T @ weighted_A) / len(self.batch_used)

    def proximal(self, x: Array, penalty: float, **kwargs: Any) -> Array:  # noqa: ARG002, ANN401
        """
        Proximal at x solved using an iterative method.

        The proximal for logistic regression does not have closed form solution, will use
        a gradient based approximation method over the entire dataset, over at most 100 iterations.

        See
        :meth:`Cost.proximal() <decent_bench.costs.Cost.proximal>`
        for the general proximal definition.

        """
        prev_batch_size = self.batch_size
        self._batch_size = self.n_samples  # Use full dataset for proximal
        approx = ca.proximal_solver(self, x, penalty)
        self._batch_size = prev_batch_size  # Restore previous batch size
        return approx

    def _get_batch_data(
        self,
        indices: EmpiricalRiskIndices = "batch",
    ) -> tuple[Array, Array]:
        """Get feature matrix and target vector for the selected samples."""
        indices = self._sample_batch_indices(indices)

        if len(indices) == self.n_samples:
            if self.A is None or self.b is None:
                self.A = iop.stack([x for x, _ in self._dataset])
                targets = iop.stack([y for _, y in self._dataset])

                targets = iop.squeeze(targets, axis=1)
                self.b = iop.where(targets == self._label_mapping[0], 0, 1)

            return self.A, self.b

        A = iop.stack([self._dataset[idx][0] for idx in indices])  # noqa: N806
        targets = iop.stack([self._dataset[idx][1] for idx in indices])

        targets = iop.squeeze(targets, axis=1)
        b = iop.where(targets == self._label_mapping[0], 0, 1)

        return A, b
