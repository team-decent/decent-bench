from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any

import numpy as np

from decent_bench.costs._base._cost import Cost
from decent_bench.utils.array import Array


class EmpiricalRiskCost(Cost, ABC):
    """Base class for empirical risk cost functions."""

    @property
    @abstractmethod
    def n_samples(self) -> int:
        """Total number of samples in dataset."""

    @property
    @abstractmethod
    def batch_size(self) -> int | None:
        """Batch size used for stochastic methods. If None, full dataset is used."""

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

    @cached_property
    def _rand(self) -> np.random.Generator:
        return np.random.default_rng(seed=0)  # Later replace with global rng

    @abstractmethod
    def predict(self, x: Array, data: list[Array]) -> Array:
        """
        Make predictions using the model parameters x on the given data.

        Returns:
            Predicted labels as an array.

        """

    @abstractmethod
    def function(self, x: Array, indices: list[int] | None = None, **kwargs: Any) -> float:  # noqa: ANN401
        """
        Evaluate function at x using datapoints at the given indices.

        If indices is None, a random batch is drawn with :attr:`batch_size` samples.
        """

    def evaluate(self, x: Array, indices: list[int] | None = None, **kwargs: Any) -> float:  # noqa: ANN401
        """Alias for :meth:`function`."""
        return self.function(x, indices=indices, **kwargs)

    def loss(self, x: Array, indices: list[int] | None = None, **kwargs: Any) -> float:  # noqa: ANN401
        """Alias for :meth:`function`."""
        return self.function(x, indices=indices, **kwargs)

    def f(self, x: Array, indices: list[int] | None = None, **kwargs: Any) -> float:  # noqa: ANN401
        """Alias for :meth:`function`."""
        return self.function(x, indices=indices, **kwargs)

    @abstractmethod
    def gradient(self, x: Array, indices: list[int] | None = None, **kwargs: Any) -> Array:  # noqa: ANN401
        """
        Gradient at x using datapoints at the given indices.

        If indices is None, a random batch is drawn with :attr:`batch_size` samples.
        """

    @abstractmethod
    def hessian(self, x: Array, indices: list[int] | None = None, **kwargs: Any) -> Array:  # noqa: ANN401
        """
        Hessian at x using datapoints at the given indices.

        If indices is None, a random batch is drawn with :attr:`batch_size` samples.
        """

    @abstractmethod
    def proximal(self, x: Array, rho: float, indices: list[int] | None = None, **kwargs: Any) -> Array:  # noqa: ANN401
        r"""
        Proximal at x using datapoints at the given indices.

        If indices is None, a random batch is drawn with :attr:`batch_size` samples.

        The proximal operator is defined as:

        .. include:: snippets/proximal_operator.rst

        If the cost function's proximal does not have a closed form solution, it can be solved iteratively using
        :meth:`~decent_bench.centralized_algorithms.proximal_solver`.
        """

    def _sample_batch_indices(self) -> list[int]:
        """
        Sample a batch of indices uniformly without replacement.

        This method uses :attr:`batch_size` to determine the size of the batch. Once a batch is sampled, it is also
        stored in :attr:`batch_used` for later reference.

        Override this method for custom sampling strategies.

        Returns:
            List of sampled indices.

        """
        if self.batch_size is not None and self.batch_size < self.n_samples:
            # Sample a random batch
            self._last_batch_used = self._rand.choice(self.n_samples, size=self.batch_size, replace=False).tolist()
        else:
            # Use full dataset
            self._last_batch_used = list(range(self.n_samples))

        return self._last_batch_used

    @abstractmethod
    def _get_batch_data(self, indices: list[int]) -> Any:  # noqa: ANN401
        """Get training data corresponding to the given batch indices."""
