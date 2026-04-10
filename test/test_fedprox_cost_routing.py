from typing import Any

import numpy as np

from decent_bench.agents import Agent
from decent_bench.costs import BaseRegularizerCost, Cost, EmpiricalRiskCost, ZeroCost
from decent_bench.distributed_algorithms import FedProx
from decent_bench.utils.types import (
    Dataset,
    EmpiricalRiskIndices,
    EmpiricalRiskReduction,
    SupportedDevices,
    SupportedFrameworks,
)


class TrackingCost(Cost):
    def __init__(self, gradient_value: float = 1.0):
        self.gradient_kwargs: list[dict[str, Any]] = []
        self._gradient = np.array([gradient_value], dtype=float)

    @property
    def shape(self) -> tuple[int, ...]:
        return (1,)

    @property
    def framework(self) -> SupportedFrameworks:
        return SupportedFrameworks.NUMPY

    @property
    def device(self) -> SupportedDevices:
        return SupportedDevices.CPU

    @property
    def m_smooth(self) -> float:
        return 0.0

    @property
    def m_cvx(self) -> float:
        return 0.0

    def function(self, x: np.ndarray, **kwargs: Any) -> float:
        del x, kwargs
        return 0.0

    def gradient(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        del x
        self.gradient_kwargs.append(dict(kwargs))
        return self._gradient.copy()

    def hessian(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        del x, kwargs
        return np.zeros((1, 1), dtype=float)

    def proximal(self, x: np.ndarray, rho: float, **kwargs: Any) -> np.ndarray:
        del rho, kwargs
        return x


class TrackingRegularizerCost(BaseRegularizerCost):
    def __init__(self, gradient_value: float = 0.0):
        super().__init__(shape=(1,))
        self.gradient_kwargs: list[dict[str, Any]] = []
        self._gradient = np.array([gradient_value], dtype=float)

    @property
    def m_smooth(self) -> float:
        return 0.0

    @property
    def m_cvx(self) -> float:
        return 0.0

    def function(self, x: np.ndarray, **kwargs: Any) -> float:
        del x, kwargs
        return 0.0

    def gradient(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        del x
        self.gradient_kwargs.append(dict(kwargs))
        return self._gradient.copy()

    def hessian(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        del x, kwargs
        return np.zeros((1, 1), dtype=float)

    def proximal(self, x: np.ndarray, rho: float, **kwargs: Any) -> np.ndarray:
        del rho, kwargs
        return x


class TrackingEmpiricalCost(EmpiricalRiskCost):
    def __init__(self, n_samples: int = 5, batch_size: int = 2, gradient_value: float = 1.0):
        self._dataset: Dataset = [(np.array([float(i)]), np.array([0.0])) for i in range(n_samples)]
        self._batch_size = batch_size
        self.gradient_indices: list[list[int]] = []
        self._gradient = np.array([gradient_value], dtype=float)

    @property
    def shape(self) -> tuple[int, ...]:
        return (1,)

    @property
    def framework(self) -> SupportedFrameworks:
        return SupportedFrameworks.NUMPY

    @property
    def device(self) -> SupportedDevices:
        return SupportedDevices.CPU

    @property
    def m_smooth(self) -> float:
        return 0.0

    @property
    def m_cvx(self) -> float:
        return 0.0

    @property
    def n_samples(self) -> int:
        return len(self._dataset)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    def predict(self, x: np.ndarray, data: list[np.ndarray]) -> np.ndarray:
        del x
        return np.asarray(data)

    def function(self, x: np.ndarray, indices: EmpiricalRiskIndices = "batch", **kwargs: Any) -> float:
        del x, kwargs
        self._sample_batch_indices(indices)
        return 0.0

    def gradient(
        self,
        x: np.ndarray,
        indices: EmpiricalRiskIndices = "batch",
        reduction: EmpiricalRiskReduction = "mean",
        **kwargs: Any,
    ) -> np.ndarray:
        del x, kwargs
        sampled_indices = self._sample_batch_indices(indices)
        self.gradient_indices.append(list(sampled_indices))
        if reduction is None:
            return np.repeat(self._gradient[np.newaxis, :], len(sampled_indices), axis=0)
        return self._gradient.copy()

    def hessian(self, x: np.ndarray, indices: EmpiricalRiskIndices = "batch", **kwargs: Any) -> np.ndarray:
        del x, kwargs
        self._sample_batch_indices(indices)
        return np.zeros((1, 1), dtype=float)

    def _get_batch_data(self, indices: EmpiricalRiskIndices = "batch") -> list[tuple[np.ndarray, np.ndarray]]:
        sampled_indices = self._sample_batch_indices(indices)
        return [self._dataset[i] for i in sampled_indices]


def _run_fedprox_local_update(
    cost: Cost,
    *,
    step_size: float = 1.0,
    num_local_epochs: int = 1,
    mu: float = 0.5,
) -> np.ndarray:
    algorithm = FedProx(iterations=1, step_size=step_size, num_local_epochs=num_local_epochs, mu=mu)
    client = Agent(0, cost)
    server = Agent(1, ZeroCost(cost.shape))
    client.initialize(x=np.zeros(cost.shape, dtype=float))
    server.initialize(x=np.zeros(cost.shape, dtype=float))
    return algorithm._compute_local_update(client, server)


def test_empirical_costs_use_minibatch_local_updates_in_fedprox() -> None:
    cost = TrackingEmpiricalCost(n_samples=5, batch_size=2)

    updated = _run_fedprox_local_update(cost, num_local_epochs=3)

    np.testing.assert_allclose(updated, np.array([-1.75]))
    assert len(cost.gradient_indices) == 3
    assert all(len(indices) == 2 for indices in cost.gradient_indices)
    assert set().union(*(set(indices) for indices in cost.gradient_indices)) == set(range(cost.n_samples))


def test_empirical_regularized_costs_keep_minibatch_local_updates_in_fedprox() -> None:
    empirical_cost = TrackingEmpiricalCost(n_samples=5, batch_size=2)
    regularizer = TrackingRegularizerCost()
    objective = empirical_cost + regularizer

    updated = _run_fedprox_local_update(objective, num_local_epochs=3)

    np.testing.assert_allclose(updated, np.array([-1.75]))
    assert len(empirical_cost.gradient_indices) == 3
    assert all(len(indices) == 2 for indices in empirical_cost.gradient_indices)
    assert set().union(*(set(indices) for indices in empirical_cost.gradient_indices)) == set(
        range(empirical_cost.n_samples)
    )
    assert len(regularizer.gradient_kwargs) == 3
    assert all(kwargs == {} for kwargs in regularizer.gradient_kwargs)


def test_scaled_empirical_costs_keep_minibatch_local_updates_in_fedprox() -> None:
    empirical_cost = TrackingEmpiricalCost(n_samples=5, batch_size=2)
    objective = 2.0 * empirical_cost

    updated = _run_fedprox_local_update(objective, num_local_epochs=3)

    np.testing.assert_allclose(updated, np.array([-3.5]))
    assert len(empirical_cost.gradient_indices) == 3
    assert all(len(indices) == 2 for indices in empirical_cost.gradient_indices)
    assert set().union(*(set(indices) for indices in empirical_cost.gradient_indices)) == set(
        range(empirical_cost.n_samples)
    )


def test_plain_costs_use_full_gradient_local_updates_in_fedprox() -> None:
    cost = TrackingCost(gradient_value=1.0)

    updated = _run_fedprox_local_update(cost, num_local_epochs=3)

    np.testing.assert_allclose(updated, np.array([-1.75]))
    assert len(cost.gradient_kwargs) == 3
    assert all(kwargs == {} for kwargs in cost.gradient_kwargs)


def test_scaled_costs_over_non_empirical_terms_use_full_gradient_local_updates_in_fedprox() -> None:
    cost = TrackingCost(gradient_value=1.0)
    objective = 2.0 * cost

    updated = _run_fedprox_local_update(objective, num_local_epochs=3)

    np.testing.assert_allclose(updated, np.array([-3.5]))
    assert len(cost.gradient_kwargs) == 3
    assert all(kwargs == {} for kwargs in cost.gradient_kwargs)


def test_regularizers_follow_the_non_batched_local_update_path_in_fedprox() -> None:
    regularizer = TrackingRegularizerCost(gradient_value=1.0)

    updated = _run_fedprox_local_update(regularizer, num_local_epochs=3)

    np.testing.assert_allclose(updated, np.array([-1.75]))
    assert len(regularizer.gradient_kwargs) == 3
    assert all(kwargs == {} for kwargs in regularizer.gradient_kwargs)
