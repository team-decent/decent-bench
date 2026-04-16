from typing import Any

import numpy as np
import pytest

from decent_bench.agents import Agent
from decent_bench.costs import BaseRegularizerCost, Cost, EmpiricalRiskCost, ZeroCost
from decent_bench.distributed_algorithms import FedAvg, FedProx, Scaffold
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


class TrackingZeroCost(ZeroCost):
    def __init__(self):
        super().__init__(shape=(1,))
        self.gradient_kwargs: list[dict[str, Any]] = []

    def gradient(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        self.gradient_kwargs.append(dict(kwargs))
        return super().gradient(x, **kwargs)


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


def _run_federated_local_update(
    algorithm_name: str,
    cost: Cost,
    *,
    step_size: float = 1.0,
    num_local_epochs: int = 1,
    mu: float = 0.5,
) -> np.ndarray:
    if algorithm_name == "fedavg":
        algorithm = FedAvg(iterations=1, step_size=step_size, num_local_epochs=num_local_epochs)
    elif algorithm_name == "fedprox":
        algorithm = FedProx(iterations=1, step_size=step_size, num_local_epochs=num_local_epochs, mu=mu)
    elif algorithm_name == "scaffold":
        algorithm = Scaffold(iterations=1, step_size=step_size, num_local_epochs=num_local_epochs)
    else:
        raise ValueError(f"Unsupported federated algorithm: {algorithm_name}")

    client = Agent(0, cost)
    server = Agent(1, ZeroCost(cost.shape))
    aux_vars = None
    if isinstance(algorithm, Scaffold):
        aux_vars = {
            "c_i": np.zeros(cost.shape, dtype=float),
            "c": np.zeros(cost.shape, dtype=float),
        }
    client.initialize(x=np.zeros(cost.shape, dtype=float), aux_vars=aux_vars)
    server.initialize(x=np.zeros(cost.shape, dtype=float))
    local_update = algorithm._compute_local_update(client, server)
    if isinstance(algorithm, Scaffold):
        return local_update[0]
    return local_update


@pytest.mark.parametrize(
    ("algorithm_name", "expected"),
    [
        pytest.param("fedavg", -3.0, id="fedavg"),
        pytest.param("fedprox", -1.75, id="fedprox"),
        pytest.param("scaffold", -3.0, id="scaffold"),
    ],
)
def test_empirical_costs_use_minibatch_local_updates(algorithm_name: str, expected: float) -> None:
    cost = TrackingEmpiricalCost(n_samples=5, batch_size=2)

    updated = _run_federated_local_update(algorithm_name, cost, num_local_epochs=3)

    np.testing.assert_allclose(updated, np.array([expected]))
    assert len(cost.gradient_indices) == 3
    assert all(len(indices) == 2 for indices in cost.gradient_indices)
    assert set().union(*(set(indices) for indices in cost.gradient_indices)) == set(range(cost.n_samples))


@pytest.mark.parametrize(
    ("algorithm_name", "expected"),
    [
        pytest.param("fedavg", -3.0, id="fedavg"),
        pytest.param("fedprox", -1.75, id="fedprox"),
        pytest.param("scaffold", -3.0, id="scaffold"),
    ],
)
def test_empirical_regularized_costs_keep_minibatch_local_updates(algorithm_name: str, expected: float) -> None:
    empirical_cost = TrackingEmpiricalCost(n_samples=5, batch_size=2)
    regularizer = TrackingRegularizerCost()
    objective = empirical_cost + regularizer

    updated = _run_federated_local_update(algorithm_name, objective, num_local_epochs=3)

    np.testing.assert_allclose(updated, np.array([expected]))
    assert len(empirical_cost.gradient_indices) == 3
    assert all(len(indices) == 2 for indices in empirical_cost.gradient_indices)
    assert set().union(*(set(indices) for indices in empirical_cost.gradient_indices)) == set(
        range(empirical_cost.n_samples)
    )
    assert len(regularizer.gradient_kwargs) == 3
    assert all(kwargs == {} for kwargs in regularizer.gradient_kwargs)


@pytest.mark.parametrize(
    ("algorithm_name", "expected"),
    [
        pytest.param("fedavg", -6.0, id="fedavg"),
        pytest.param("fedprox", -3.5, id="fedprox"),
        pytest.param("scaffold", -6.0, id="scaffold"),
    ],
)
def test_scaled_empirical_costs_keep_minibatch_local_updates(algorithm_name: str, expected: float) -> None:
    empirical_cost = TrackingEmpiricalCost(n_samples=5, batch_size=2)
    objective = 2.0 * empirical_cost

    updated = _run_federated_local_update(algorithm_name, objective, num_local_epochs=3)

    np.testing.assert_allclose(updated, np.array([expected]))
    assert len(empirical_cost.gradient_indices) == 3
    assert all(len(indices) == 2 for indices in empirical_cost.gradient_indices)
    assert set().union(*(set(indices) for indices in empirical_cost.gradient_indices)) == set(
        range(empirical_cost.n_samples)
    )


@pytest.mark.parametrize(
    ("algorithm_name", "expected"),
    [
        pytest.param("fedavg", -3.0, id="fedavg"),
        pytest.param("fedprox", -1.75, id="fedprox"),
        pytest.param("scaffold", -3.0, id="scaffold"),
    ],
)
def test_plain_costs_use_full_gradient_local_updates(algorithm_name: str, expected: float) -> None:
    cost = TrackingCost(gradient_value=1.0)

    updated = _run_federated_local_update(algorithm_name, cost, num_local_epochs=3)

    np.testing.assert_allclose(updated, np.array([expected]))
    assert len(cost.gradient_kwargs) == 3
    assert all(kwargs == {} for kwargs in cost.gradient_kwargs)


@pytest.mark.parametrize("algorithm_name", ["fedavg", "scaffold"])
def test_sum_costs_over_non_empirical_terms_use_full_gradient_local_updates(algorithm_name: str) -> None:
    cost_a = TrackingCost(gradient_value=1.0)
    cost_b = TrackingCost(gradient_value=2.0)
    objective = cost_a + cost_b

    updated = _run_federated_local_update(algorithm_name, objective, num_local_epochs=2)

    np.testing.assert_allclose(updated, np.array([-6.0]))
    assert len(cost_a.gradient_kwargs) == 2
    assert len(cost_b.gradient_kwargs) == 2
    assert all(kwargs == {} for kwargs in cost_a.gradient_kwargs)
    assert all(kwargs == {} for kwargs in cost_b.gradient_kwargs)


@pytest.mark.parametrize(
    ("algorithm_name", "expected", "num_local_epochs"),
    [
        pytest.param("fedavg", -4.0, 2, id="fedavg"),
        pytest.param("fedprox", -3.5, 3, id="fedprox"),
        pytest.param("scaffold", -4.0, 2, id="scaffold"),
    ],
)
def test_scaled_costs_over_non_empirical_terms_use_full_gradient_local_updates(
    algorithm_name: str, expected: float, num_local_epochs: int
) -> None:
    cost = TrackingCost(gradient_value=1.0)
    objective = 2.0 * cost

    updated = _run_federated_local_update(algorithm_name, objective, num_local_epochs=num_local_epochs)

    np.testing.assert_allclose(updated, np.array([expected]))
    assert len(cost.gradient_kwargs) == num_local_epochs
    assert all(kwargs == {} for kwargs in cost.gradient_kwargs)


@pytest.mark.parametrize(
    ("algorithm_name", "expected", "num_local_epochs"),
    [
        pytest.param("fedavg", -2.0, 2, id="fedavg"),
        pytest.param("fedprox", -1.75, 3, id="fedprox"),
        pytest.param("scaffold", -2.0, 2, id="scaffold"),
    ],
)
def test_regularizers_follow_the_non_batched_local_update_path(
    algorithm_name: str, expected: float, num_local_epochs: int
) -> None:
    regularizer = TrackingRegularizerCost(gradient_value=1.0)

    updated = _run_federated_local_update(algorithm_name, regularizer, num_local_epochs=num_local_epochs)

    np.testing.assert_allclose(updated, np.array([expected]))
    assert len(regularizer.gradient_kwargs) == num_local_epochs
    assert all(kwargs == {} for kwargs in regularizer.gradient_kwargs)


@pytest.mark.parametrize("algorithm_name", ["fedavg", "scaffold"])
def test_zero_costs_do_not_need_special_local_update_handling(algorithm_name: str) -> None:
    cost = TrackingZeroCost()

    updated = _run_federated_local_update(algorithm_name, cost, num_local_epochs=3)

    np.testing.assert_allclose(updated, np.array([0.0]))
    assert len(cost.gradient_kwargs) == 3
    assert all(kwargs == {} for kwargs in cost.gradient_kwargs)
