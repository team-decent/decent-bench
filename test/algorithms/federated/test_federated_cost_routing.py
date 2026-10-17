from typing import Any

import numpy as np
import pytest

from decent_array.types import Devices, Frameworks
from decent_array import Array, interoperability as iop

from decent_bench.agents import Agent
from decent_bench.algorithms.federated import FedAdagrad, FedAdam, FedAvg, FedDyn, FedPD, FedProx, FedYogi, Scaffold
from decent_bench.costs import BaseRegularizerCost, Cost, EmpiricalRiskCost, ZeroCost
from decent_bench.utils.types import (
    Dataset,
    EmpiricalRiskIndices,
    EmpiricalRiskReduction,
)
from decent_bench.costs._decorators import autodecorate_cost_method


class TrackingCost(Cost):
    def __init__(self, gradient_value: float = 1.0):
        self.gradient_kwargs: list[dict[str, Any]] = []
        self._gradient = np.array([gradient_value], dtype=float)

    @property
    def shape(self) -> tuple[int, ...]:
        return (1,)

    @property
    def framework(self) -> Frameworks:
        return Frameworks.NUMPY

    @property
    def device(self) -> Devices:
        return Devices.CPU

    @property
    def m_smooth(self) -> float:
        return 0.0

    @property
    def m_cvx(self) -> float:
        return 0.0

    @autodecorate_cost_method(Cost.function)
    def function(self, x: np.ndarray, **kwargs: Any) -> float:
        del x, kwargs
        return 0.0

    @autodecorate_cost_method(Cost.gradient)
    def gradient(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        del x
        self.gradient_kwargs.append(dict(kwargs))
        return self._gradient.copy()

    @autodecorate_cost_method(Cost.hessian)
    def hessian(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        del x, kwargs
        return np.zeros((1, 1), dtype=float)

    @autodecorate_cost_method(Cost.proximal)
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

    @autodecorate_cost_method(Cost.function)
    def function(self, x: np.ndarray, **kwargs: Any) -> float:
        del x, kwargs
        return 0.0

    @autodecorate_cost_method(Cost.gradient)
    def gradient(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        del x
        self.gradient_kwargs.append(dict(kwargs))
        return self._gradient.copy()

    @autodecorate_cost_method(Cost.hessian)
    def hessian(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        del x, kwargs
        return np.zeros((1, 1), dtype=float)

    @autodecorate_cost_method(Cost.proximal)
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
    def framework(self) -> Frameworks:
        return Frameworks.NUMPY

    @property
    def device(self) -> Devices:
        return Devices.CPU

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

    @autodecorate_cost_method(EmpiricalRiskCost.predict)
    def predict(self, x: np.ndarray, data: list[np.ndarray]) -> np.ndarray:
        del x
        return np.asarray(data)

    @autodecorate_cost_method(Cost.function)
    def function(self, x: np.ndarray, indices: EmpiricalRiskIndices = "batch", **kwargs: Any) -> float:
        del x, kwargs
        self._sample_batch_indices(indices)
        return 0.0

    @autodecorate_cost_method(Cost.gradient)
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

    @autodecorate_cost_method(Cost.hessian)
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
    num_local_steps: int = 1,
    mu: float = 0.5,
) -> Array:
    client = Agent(cost)
    server = Agent(ZeroCost(cost.shape))

    if algorithm_name == "fedavg":
        algorithm = FedAvg(step_size=step_size, num_local_steps=num_local_steps)
    elif algorithm_name == "fedadagrad":
        algorithm = FedAdagrad(step_size=step_size, num_local_steps=num_local_steps)
    elif algorithm_name == "fedyogi":
        algorithm = FedYogi(step_size=step_size, num_local_steps=num_local_steps)
    elif algorithm_name == "fedadam":
        algorithm = FedAdam(step_size=step_size, num_local_steps=num_local_steps)
    elif algorithm_name == "fedprox":
        algorithm = FedProx(step_size=step_size, num_local_steps=num_local_steps, penalty=mu)
    elif algorithm_name == "feddyn":
        algorithm = FedDyn(step_size=step_size, penalty=1.0, num_local_steps=num_local_steps)
    elif algorithm_name == "fedpd":
        algorithm = FedPD(step_size=step_size, penalty=1.0, num_local_steps=num_local_steps)
    elif algorithm_name == "scaffold":
        algorithm = Scaffold(step_size=step_size, num_local_steps=num_local_steps)
    else:
        raise ValueError(f"Unsupported federated algorithm: {algorithm_name}")

    aux_vars = None
    if isinstance(algorithm, Scaffold):
        aux_vars = {"c_i": iop.zeros(cost.shape)}
    elif isinstance(algorithm, FedDyn):
        aux_vars = {"g": iop.zeros(cost.shape)}
    elif isinstance(algorithm, FedPD):
        aux_vars = {
            "lambda": iop.zeros(cost.shape),
            "center": iop.zeros(cost.shape),
        }
    client.initialize(x=iop.zeros(cost.shape), aux_vars=aux_vars)
    server.initialize(x=iop.zeros(cost.shape))
    if isinstance(algorithm, Scaffold):
        client._received_messages.put(  # noqa: SLF001
            server,
            iop.stack([iop.zeros(cost.shape), iop.zeros(cost.shape)]),
        )
    elif not isinstance(algorithm, FedPD):
        client._received_messages.put(server, iop.zeros(cost.shape))  # noqa: SLF001
    if isinstance(algorithm, FedPD):
        algorithm._num_local_steps_by_client = {client: num_local_steps}  # noqa: SLF001
    if isinstance(algorithm, FedPD):
        local_update = algorithm._compute_local_update(client)
    elif isinstance(algorithm, FedDyn):
        local_update = algorithm._compute_local_update(client, client.message(server))
    else:
        local_update = algorithm._compute_local_update(client, server)
    if isinstance(algorithm, Scaffold):
        return local_update[0]
    return local_update


@pytest.mark.parametrize(
    ("algorithm_name", "expected"),
    [
        pytest.param("fedavg", -3.0, id="fedavg"),
        pytest.param("fedadagrad", -3.0, id="fedadagrad"),
        pytest.param("fedyogi", -3.0, id="fedyogi"),
        pytest.param("fedadam", -3.0, id="fedadam"),
        pytest.param("fedprox", -1.75, id="fedprox"),
        pytest.param("feddyn", -1.0, id="feddyn"),
        pytest.param("fedpd", -1.0, id="fedpd"),
        pytest.param("scaffold", -3.0, id="scaffold"),
    ],
)
def test_empirical_costs_use_minibatch_local_updates(algorithm_name: str, expected: float) -> None:
    cost = TrackingEmpiricalCost(n_samples=5, batch_size=2)

    updated = _run_federated_local_update(algorithm_name, cost, num_local_steps=3)

    np.testing.assert_allclose(updated.value, np.array([expected]))
    assert len(cost.gradient_indices) == 3
    assert all(len(indices) == 2 for indices in cost.gradient_indices)
    assert set().union(*(set(indices) for indices in cost.gradient_indices)) == set(range(cost.n_samples))


@pytest.mark.parametrize(
    ("algorithm_name", "expected"),
    [
        pytest.param("fedavg", -3.0, id="fedavg"),
        pytest.param("fedadagrad", -3.0, id="fedadagrad"),
        pytest.param("fedyogi", -3.0, id="fedyogi"),
        pytest.param("fedadam", -3.0, id="fedadam"),
        pytest.param("fedprox", -1.75, id="fedprox"),
        pytest.param("feddyn", -1.0, id="feddyn"),
        pytest.param("fedpd", -1.0, id="fedpd"),
        pytest.param("scaffold", -3.0, id="scaffold"),
    ],
)
def test_empirical_regularized_costs_keep_minibatch_local_updates(algorithm_name: str, expected: float) -> None:
    empirical_cost = TrackingEmpiricalCost(n_samples=5, batch_size=2)
    regularizer = TrackingRegularizerCost()
    objective = empirical_cost + regularizer

    updated = _run_federated_local_update(algorithm_name, objective, num_local_steps=3)

    np.testing.assert_allclose(updated.value, np.array([expected]))
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
        pytest.param("fedadagrad", -6.0, id="fedadagrad"),
        pytest.param("fedyogi", -6.0, id="fedyogi"),
        pytest.param("fedadam", -6.0, id="fedadam"),
        pytest.param("fedprox", -3.5, id="fedprox"),
        pytest.param("feddyn", -2.0, id="feddyn"),
        pytest.param("fedpd", -2.0, id="fedpd"),
        pytest.param("scaffold", -6.0, id="scaffold"),
    ],
)
def test_scaled_empirical_costs_keep_minibatch_local_updates(algorithm_name: str, expected: float) -> None:
    empirical_cost = TrackingEmpiricalCost(n_samples=5, batch_size=2)
    objective = 2.0 * empirical_cost

    updated = _run_federated_local_update(algorithm_name, objective, num_local_steps=3)

    np.testing.assert_allclose(updated.value, np.array([expected]))
    assert len(empirical_cost.gradient_indices) == 3
    assert all(len(indices) == 2 for indices in empirical_cost.gradient_indices)
    assert set().union(*(set(indices) for indices in empirical_cost.gradient_indices)) == set(
        range(empirical_cost.n_samples)
    )


@pytest.mark.parametrize(
    ("algorithm_name", "expected"),
    [
        pytest.param("fedavg", -3.0, id="fedavg"),
        pytest.param("fedadagrad", -3.0, id="fedadagrad"),
        pytest.param("fedyogi", -3.0, id="fedyogi"),
        pytest.param("fedadam", -3.0, id="fedadam"),
        pytest.param("fedprox", -1.75, id="fedprox"),
        pytest.param("feddyn", -1.0, id="feddyn"),
        pytest.param("fedpd", -1.0, id="fedpd"),
        pytest.param("scaffold", -3.0, id="scaffold"),
    ],
)
def test_plain_costs_use_full_gradient_local_updates(algorithm_name: str, expected: float) -> None:
    cost = TrackingCost(gradient_value=1.0)

    updated = _run_federated_local_update(algorithm_name, cost, num_local_steps=3)

    np.testing.assert_allclose(updated.value, np.array([expected]))
    assert len(cost.gradient_kwargs) == 3
    assert all(kwargs == {} for kwargs in cost.gradient_kwargs)


@pytest.mark.parametrize(
    "algorithm_name", ["fedavg", "fedadagrad", "fedyogi", "fedadam", "feddyn", "fedpd", "scaffold"]
)
def test_sum_costs_over_non_empirical_terms_use_full_gradient_local_updates(algorithm_name: str) -> None:
    cost_a = TrackingCost(gradient_value=1.0)
    cost_b = TrackingCost(gradient_value=2.0)
    objective = cost_a + cost_b

    updated = _run_federated_local_update(algorithm_name, objective, num_local_steps=2)

    expected = -3.0 if algorithm_name in {"feddyn", "fedpd"} else -6.0
    np.testing.assert_allclose(updated.value, np.array([expected]))
    assert len(cost_a.gradient_kwargs) == 2
    assert len(cost_b.gradient_kwargs) == 2
    assert all(kwargs == {} for kwargs in cost_a.gradient_kwargs)
    assert all(kwargs == {} for kwargs in cost_b.gradient_kwargs)


@pytest.mark.parametrize(
    ("algorithm_name", "expected", "num_local_steps"),
    [
        pytest.param("fedavg", -4.0, 2, id="fedavg"),
        pytest.param("fedadagrad", -4.0, 2, id="fedadagrad"),
        pytest.param("fedyogi", -4.0, 2, id="fedyogi"),
        pytest.param("fedadam", -4.0, 2, id="fedadam"),
        pytest.param("fedprox", -3.5, 3, id="fedprox"),
        pytest.param("feddyn", -2.0, 2, id="feddyn"),
        pytest.param("fedpd", -2.0, 2, id="fedpd"),
        pytest.param("scaffold", -4.0, 2, id="scaffold"),
    ],
)
def test_scaled_costs_over_non_empirical_terms_use_full_gradient_local_updates(
    algorithm_name: str, expected: float, num_local_steps: int
) -> None:
    cost = TrackingCost(gradient_value=1.0)
    objective = 2.0 * cost

    updated = _run_federated_local_update(algorithm_name, objective, num_local_steps=num_local_steps)

    np.testing.assert_allclose(updated.value, np.array([expected]))
    assert len(cost.gradient_kwargs) == num_local_steps
    assert all(kwargs == {} for kwargs in cost.gradient_kwargs)


@pytest.mark.parametrize(
    ("algorithm_name", "expected", "num_local_steps"),
    [
        pytest.param("fedavg", -2.0, 2, id="fedavg"),
        pytest.param("fedadagrad", -2.0, 2, id="fedadagrad"),
        pytest.param("fedyogi", -2.0, 2, id="fedyogi"),
        pytest.param("fedadam", -2.0, 2, id="fedadam"),
        pytest.param("fedprox", -1.75, 3, id="fedprox"),
        pytest.param("feddyn", -1.0, 2, id="feddyn"),
        pytest.param("fedpd", -1.0, 2, id="fedpd"),
        pytest.param("scaffold", -2.0, 2, id="scaffold"),
    ],
)
def test_regularizers_follow_the_non_batched_local_update_path(
    algorithm_name: str, expected: float, num_local_steps: int
) -> None:
    regularizer = TrackingRegularizerCost(gradient_value=1.0)

    updated = _run_federated_local_update(algorithm_name, regularizer, num_local_steps=num_local_steps)

    np.testing.assert_allclose(updated.value, np.array([expected]))
    assert len(regularizer.gradient_kwargs) == num_local_steps
    assert all(kwargs == {} for kwargs in regularizer.gradient_kwargs)


@pytest.mark.parametrize(
    "algorithm_name", ["fedavg", "fedadagrad", "fedyogi", "fedadam", "feddyn", "fedpd", "scaffold"]
)
def test_zero_costs_do_not_need_special_local_update_handling(algorithm_name: str) -> None:
    cost = TrackingZeroCost()

    updated = _run_federated_local_update(algorithm_name, cost, num_local_steps=3)

    np.testing.assert_allclose(updated.value, np.array([0.0]))
    assert len(cost.gradient_kwargs) == 3
    assert all(kwargs == {} for kwargs in cost.gradient_kwargs)
