from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any

import numpy as np

import decent_bench.utils.interoperability as iop
from decent_bench.costs._base._cost import Cost
from decent_bench.utils.array import Array
from decent_bench.utils.types import Dataset, EmpiricalRiskIndices, EmpiricalRiskReduction


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
            - int: datapoint to use.
            - list[int]: datapoints to use.
            - "all": use the full dataset.
            - "batch": draw a batch with :attr:`batch_size` samples.

        """

    def __mul__(self, other: float) -> Cost:
        """
        Multiply by a scalar while preserving the empirical-risk abstraction.

        Raises:
            TypeError: If other is not a real scalar.

        """
        if not self._is_valid_scalar(other):
            raise TypeError(f"Cost can only be multiplied by a real number, got {type(other)}.")
        from decent_bench.costs._empirical_risk._empirical_scaled_cost import _EmpiricalScaledCost  # noqa: PLC0415

        return _EmpiricalScaledCost(self, float(other))

    def __truediv__(self, other: float) -> Cost:
        """
        Divide by a scalar while preserving the empirical-risk abstraction.

        Raises:
            TypeError: If other is not a real scalar.
            ZeroDivisionError: If other is zero.

        """
        if not self._is_valid_scalar(other):
            raise TypeError(f"Cost can only be divided by a real number, got {type(other)}.")
        if other == 0:
            raise ZeroDivisionError("Division by zero is not allowed for Cost objects.")
        return self.__mul__(1.0 / float(other))

    def __neg__(self) -> Cost:
        """Negate this empirical risk while preserving the empirical-risk abstraction."""
        return self.__mul__(-1.0)

    def __add__(self, other: Cost) -> Cost:
        """Add another cost, preserving the empirical-risk abstraction for regularization."""
        self._validate_cost_operation(other)
        from decent_bench.costs._base._regularizer_costs import BaseRegularizerCost  # noqa: PLC0415

        if isinstance(other, BaseRegularizerCost):
            from decent_bench.costs._empirical_risk._empirical_regularized_cost import (  # noqa: PLC0415
                EmpiricalRegularizedCost,
            )

            return EmpiricalRegularizedCost(self, other)
        return super().__add__(other)

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
    def gradient(
        self,
        x: Array,
        indices: EmpiricalRiskIndices = "batch",
        reduction: EmpiricalRiskReduction = "mean",
        **kwargs: Any,  # noqa: ANN401
    ) -> Array:
        """
        Gradient at x using datapoints at the given indices.

        The returned gradient is the mean of per-sample gradients over the selected samples.

        Supported values for indices are:
            - int: datapoint to use.
            - list[int]: datapoints to use.
            - "all": use the full dataset.
            - "batch": draw a batch with :attr:`batch_size` samples.

        Supported values for reduction are:
            - "mean": average the gradients over the samples.
            - None: return the gradients for each sample, index as the first dimension.

        Note:
            When reduction is None, the returned array will have an additional leading dimension
            corresponding to the number of samples used. Indexing into this dimension will give the gradient
            for the respective sample in :attr:`batch_used <decent_bench.costs.EmpiricalRiskCost.batch_used>`.

        """

    @abstractmethod
    def hessian(self, x: Array, indices: EmpiricalRiskIndices = "batch", **kwargs: Any) -> Array:  # noqa: ANN401
        """
        Hessian at x using datapoints at the given indices.

        The returned Hessian is the mean of per-sample Hessians over the selected samples.

        Supported values for indices are:
            - int: datapoint to use.
            - list[int]: datapoints to use.
            - "all": use the full dataset.
            - "batch": draw a batch with :attr:`batch_size` samples.

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
        Sample a batch of indices if indices is "batch", otherwise use the given indices.

        Supported values for indices are:
            - int: datapoint to use.
            - list[int]: datapoints to use.
            - "all": use the full dataset.
            - "batch": draw a batch with :attr:`batch_size` samples.

        This method uses :attr:`batch_size` to determine the size of the batch. For ``indices="batch"`` with
        :attr:`batch_size < n_samples <decent_bench.costs.EmpiricalRiskCost.n_samples>`, batches are sampled
        without replacement across successive calls until the full dataset is covered (epoch-style sampling).
        When there are fewer unseen indices left than :attr:`batch_size`, the remaining unseen indices are used first,
        and the rest of the batch is drawn from a newly shuffled epoch.

        Once a batch is sampled, it is also stored in :attr:`batch_used` for later reference.

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
            if self.batch_size < self.n_samples:
                remaining: list[int] = getattr(self, "_remaining_batch_indices", [])
                if len(remaining) == 0:
                    remaining = iop.get_numpy_generator().permutation(self.n_samples).tolist()

                if len(remaining) >= self.batch_size:
                    sample = remaining[: self.batch_size]
                    self._remaining_batch_indices = remaining[self.batch_size :]
                else:
                    sample = remaining
                    needed = self.batch_size - len(sample)

                    if len(sample) > 0:
                        used_now = set(sample)
                        next_epoch = (
                            iop.get_numpy_generator().permutation(list(set(range(self.n_samples)) - used_now)).tolist()
                        )
                    else:
                        next_epoch = iop.get_numpy_generator().permutation(self.n_samples).tolist()

                    sample.extend(next_epoch[:needed])
                    self._remaining_batch_indices = next_epoch[needed:]

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
            - int: datapoint to use.
            - list[int]: datapoints to use.
            - "all": use the full dataset.
            - "batch": draw a batch with :attr:`batch_size` samples.

        Make sure to call :meth:`_sample_batch_indices` (indices) to handle batch sampling and tracking.

        """
