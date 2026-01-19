from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from decent_bench.costs.base._cost import Cost
from decent_bench.utils.array import Array


class EmpiricalRiskCost(Cost, ABC):
    """Base class for empirical risk cost functions."""

    @property
    @abstractmethod
    def n_samples(self) -> int:
        """Total number of samples in dataset."""

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """Batch size used for stochastic methods."""

    @property
    @abstractmethod
    def batch_used(self) -> list[int]:
        """Indices of samples used in the most recent batch."""

    @abstractmethod
    def predict(self, x: Array, data: list[Array]) -> Array:
        """Make predictions using the model parameters x on the given data."""

    @abstractmethod
    def function(self, x: Array, indices: list[int] | None = None, **kwargs: Any) -> float:  # noqa: ANN401
        """Evaluate function at x."""

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
        """Gradient at x."""

    @abstractmethod
    def hessian(self, x: Array, indices: list[int] | None = None, **kwargs: Any) -> Array:  # noqa: ANN401
        """Hessian at x."""

    @abstractmethod
    def proximal(self, x: Array, rho: float, **kwargs: Any) -> Array:  # noqa: ANN401
        r"""
        Proximal at x.

        The proximal operator is defined as:

        .. include:: snippets/proximal_operator.rst

        If the cost function's proximal does not have a closed form solution, it can be solved iteratively using
        :meth:`~decent_bench.centralized_algorithms.proximal_solver`.
        """
