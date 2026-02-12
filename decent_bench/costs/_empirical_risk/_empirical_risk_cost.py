from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any

import numpy as np

from decent_bench.costs._base._cost import Cost
from decent_bench.utils.array import Array
from decent_bench.utils.types import Dataset, EmpiricalRiskIndices


class EmpiricalRiskCost(Cost, ABC):
    r"""
    Base class for empirical risk cost functions.

    This class provides an interface for implementing various empirical risk minimization
    problems, supporting both full-batch and mini-batch computations. This cost function class
    is designed to work with :class:`~decent_bench.utils.types.Dataset` where each
    datapoint is a tuple of (features, target), or (features, None) for unsupervised learning.

    All empirical risk values, gradients, and Hessians are defined as means over the selected
    samples (full dataset or batch), not sums.

    Mathematical Definition
    -----------------------
    Given a dataset with :math:`m` samples :math:`\{d_i\}_{i=1}^{m}`, the empirical risk is defined as:

    .. math::
        \mathcal{f}(x) = \frac{1}{m} \sum_{i=1}^{m} \ell(x, d_i)

    where:
        - :math:`x` are the model parameters
        - :math:`\ell` is the loss function measuring the discrepancy between predictions and true targets

    Stochastic Variant
    ------------------
    For large datasets, computing the full empirical risk can be expensive. Instead, a stochastic
    approximation using a mini-batch of size :math:`b < m` is often used:

    .. math::
        \mathcal{f}(x) = \frac{1}{b} \sum_{i \in \mathcal{B}} \ell(x, d_i)

    where :math:`\mathcal{B}` is a randomly sampled batch of :math:`b` indices from :math:`\{1, \ldots, m\}`.
    """

    @property
    @abstractmethod
    def n_samples(self) -> int:
        """Total number of samples in dataset."""

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """Batch size used for stochastic methods."""

    @property
    def batch_used(self) -> list[int]:
        """
        Indices of samples used in the most recent batch.

        Raises:
            ValueError: If no batch has been used yet.

        """
        if not hasattr(self, "_last_batch_used"):
            raise ValueError("No batch has been used yet.")
        return self._last_batch_used

    @property
    @abstractmethod
    def dataset(self) -> Dataset:
        """Dataset used in the empirical risk cost."""

    @cached_property
    def _rand(self) -> np.random.Generator:
        return np.random.default_rng(seed=0)  # Later replace with global rng

    @abstractmethod
    def predict(self, x: Array, data: list[Array]) -> Array:
        """
        Make predictions using the model parameters x on the given data.

        Returns:
            Predicted targets as an array.

        """

    @abstractmethod
    def function(self, x: Array, indices: EmpiricalRiskIndices = "batch", **kwargs: Any) -> float:  # noqa: ANN401
        """
        Evaluate function at x using datapoints at the given indices.

        The returned value is the mean loss over the selected samples.

        Supported values for indices are:
            - int: the corresponding datapoint is used.
            - list[int]: corresponding datapoints are used.
            - "all": the full dataset is used.
            - "batch": a batch is drawn with :attr:`batch_size` samples.
        """

    def evaluate(self, x: Array, indices: EmpiricalRiskIndices = "batch", **kwargs: Any) -> float:  # noqa: ANN401
        """Alias for :meth:`function`."""
        return self.function(x, indices=indices, **kwargs)

    def loss(self, x: Array, indices: EmpiricalRiskIndices = "batch", **kwargs: Any) -> float:  # noqa: ANN401
        """Alias for :meth:`function`."""
        return self.function(x, indices=indices, **kwargs)

    def f(self, x: Array, indices: EmpiricalRiskIndices = "batch", **kwargs: Any) -> float:  # noqa: ANN401
        """Alias for :meth:`function`."""
        return self.function(x, indices=indices, **kwargs)

    @abstractmethod
    def gradient(self, x: Array, indices: EmpiricalRiskIndices = "batch", **kwargs: Any) -> Array:  # noqa: ANN401
        """
        Gradient at x using datapoints at the given indices.

        The returned gradient is the mean of per-sample gradients over the selected samples.

        Supported values for indices are:
            - int: the corresponding datapoint is used.
            - list[int]: corresponding datapoints are used.
            - "all": the full dataset is used.
            - "batch": a batch is drawn with :attr:`batch_size` samples.
        """

    @abstractmethod
    def hessian(self, x: Array, indices: EmpiricalRiskIndices = "batch", **kwargs: Any) -> Array:  # noqa: ANN401
        """
        Hessian at x using datapoints at the given indices.

        The returned Hessian is the mean of per-sample Hessians over the selected samples.

        Supported values for indices are:
            - int: the corresponding datapoint is used.
            - list[int]: corresponding datapoints are used.
            - "all": the full dataset is used.
            - "batch": a batch is drawn with :attr:`batch_size` samples.
        """

    def proximal(self, x: Array, rho: float, **kwargs: Any) -> Array:  # noqa: ANN401
        """
        Proximal at x using the full dataset.

        The proximal operator is defined as:

        .. include:: snippets/proximal_operator.rst

        If the cost function's proximal does not have a closed form solution, it can be solved iteratively using
        :meth:`~decent_bench.centralized_algorithms.proximal_solver`.
        """
        raise NotImplementedError(
            "Proximal operator is not implemented for this cost function."
            " See centralized_algorithms.proximal_solver for an implementation of the approximate proximal computation."
        )

    def _sample_batch_indices(self, indices: EmpiricalRiskIndices = "batch") -> list[int]:
        """
        Sample a batch of indices uniformly without replacement if indices is "batch", otherwise use the given indices.

        Supported values for indices are:
            - int: the corresponding datapoint is used.
            - list[int]: corresponding datapoints are used.
            - "all": the full dataset is used.
            - "batch": a batch is drawn with :attr:`batch_size` samples.

        This method uses :attr:`batch_size` to determine the size of the batch. Once a batch is sampled, it is also
        stored in :attr:`batch_used` for later reference.

        Override this method for custom sampling strategies. Do not forget to update
        `_last_batch_used` accordingly if you override this method.

        Returns:
            List of sampled indices.

        Raises:
            ValueError: If an integer index is out of bounds.
            ValueError: If an invalid string is provided for indices.

        """
        if isinstance(indices, int):
            if indices < 0 or indices >= self.n_samples:
                raise ValueError(f"Index {indices} is out of bounds for dataset with {self.n_samples} samples.")

            self._last_batch_used = [indices]
            return self._last_batch_used

        if isinstance(indices, list):
            self._last_batch_used = indices
            return self._last_batch_used

        # It's a string
        if indices == "all":
            # Use full dataset
            self._last_batch_used = list(range(self.n_samples))
        elif indices == "batch":
            if self.batch_size is not None and self.batch_size < self.n_samples:
                # Sample a random batch
                sample: list[int] = self._rand.choice(self.n_samples, size=self.batch_size, replace=False).tolist()
                self._last_batch_used = sample
            else:
                # Use full dataset
                self._last_batch_used = list(range(self.n_samples))
        else:
            raise ValueError(f"Invalid indices string: {indices}. Only 'all' and 'batch' are supported.")

        return self._last_batch_used

    @abstractmethod
    def _get_batch_data(self, indices: EmpiricalRiskIndices = "batch") -> Any:  # noqa: ANN401
        """
        Get training data corresponding to the given batch indices.

        Supported values for indices are:
            - int: the corresponding datapoint is used.
            - list[int]: corresponding datapoints are used.
            - "all": the full dataset is used.
            - "batch": a batch is drawn with :attr:`batch_size` samples.

        Make sure to call :meth:`_sample_batch_indices` (indices) to handle batch sampling and tracking.
        """
