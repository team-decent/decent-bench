from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from decent_bench.agents import Agent
from decent_bench.costs import Cost, EmpiricalRiskCost, L1RegularizerCost, QuadraticCost, ZeroCost
from decent_bench.algorithms.federated import FedLT
from decent_bench.networks import FedNetwork
from decent_bench.schemes import ClientSelectionScheme, GaussianNoise, Quantization
from decent_bench.utils.types import (
    Dataset,
    EmpiricalRiskIndices,
    EmpiricalRiskReduction,
    SupportedDevices,
    SupportedFrameworks,
)


class ConstantGradientCost(Cost):
    def __init__(
        self,
        gradient_value: float = 1.0,
        *,
        m_smooth: float = 1.0,
        m_cvx: float = 0.0,
    ) -> None:
        self.gradient_kwargs: list[dict[str, Any]] = []
        self._gradient = np.array([gradient_value], dtype=float)
        self._m_smooth = m_smooth
        self._m_cvx = m_cvx

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
        return self._m_smooth

    @property
    def m_cvx(self) -> float:
        return self._m_cvx

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


class ShiftProxCost(ConstantGradientCost):
    def __init__(self, shift: float) -> None:
        super().__init__(gradient_value=0.0)
        self.shift = shift
        self.proximal_rhos: list[float] = []

    def proximal(self, x: np.ndarray, rho: float, **kwargs: Any) -> np.ndarray:
        del kwargs
        self.proximal_rhos.append(rho)
        return x + self.shift


class TrackingEmpiricalCost(EmpiricalRiskCost):
    def __init__(self, n_samples: int = 5, batch_size: int = 2, gradient_value: float = 1.0) -> None:
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
        return 1.0

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

    def _get_batch_data(self, indices: EmpiricalRiskIndices = "batch") -> Dataset:
        sampled_indices = self._sample_batch_indices(indices)
        return [self._dataset[i] for i in sampled_indices]


class FirstClientSelection(ClientSelectionScheme):
    def select(self, clients: Sequence[Agent], iteration: int) -> list[Agent]:
        del iteration
        return [clients[0]]


def _make_network(*costs: Cost) -> FedNetwork:
    return FedNetwork(clients=[Agent(i, cost) for i, cost in enumerate(costs)])


@pytest.mark.parametrize("use_acceleration", [False, True])
def test_fedlt_runs_with_default_and_accelerated_local_updates(use_acceleration: bool) -> None:
    network = _make_network(
        QuadraticCost(A=np.array([[1.0]]), b=np.array([-1.0])),
        QuadraticCost(A=np.array([[2.0]]), b=np.array([1.0])),
    )
    algorithm = FedLT(iterations=3, step_size=0.2, num_local_epochs=2, rho=1.0, use_acceleration=use_acceleration)

    algorithm.run(network)

    assert np.isfinite(network.server().x).all()
    assert all(np.isfinite(client.x).all() for client in network.clients())


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"rho": 0.0}, "`rho` must be positive"),
        ({"num_local_epochs": 0}, "`num_local_epochs` must be positive"),
        ({"step_size": 0.0}, "`step_size` must be positive"),
    ],
)
def test_fedlt_rejects_invalid_parameters(kwargs: dict[str, float | int], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        FedLT(iterations=1, **kwargs)


def test_fedlt_initializes_auxiliary_variables_from_z0() -> None:
    network = _make_network(ConstantGradientCost(0.0), ConstantGradientCost(0.0))
    clients = network.clients()
    z0 = {clients[0]: np.array([2.0]), clients[1]: np.array([-1.0])}
    algorithm = FedLT(iterations=1, z0=z0)

    algorithm.initialize(network)

    np.testing.assert_allclose(clients[0].aux_vars["z"], np.array([2.0]))
    np.testing.assert_allclose(clients[1].aux_vars["z"], np.array([-1.0]))
    np.testing.assert_allclose(network.server().aux_vars["z_by_client"][clients[0]], np.array([2.0]))
    np.testing.assert_allclose(network.server().aux_vars["z_by_client"][clients[1]], np.array([-1.0]))


def test_fedlt_accelerated_solver_requires_valid_client_constants() -> None:
    network = _make_network(ConstantGradientCost(m_smooth=np.inf, m_cvx=0.0))
    algorithm = FedLT(iterations=1, use_acceleration=True)

    with pytest.raises(ValueError, match="requires finite non-negative"):
        algorithm.initialize(network)


def test_fedlt_local_gradient_step_uses_penalty_term() -> None:
    client = Agent(0, ConstantGradientCost(gradient_value=1.0))
    server = Agent(1, ZeroCost((1,)))
    algorithm = FedLT(iterations=1, step_size=1.0, num_local_epochs=2, rho=1.0)
    client.initialize(x=np.array([0.0]), aux_vars={"z": np.array([0.0])})
    server.initialize(x=np.array([0.0]))
    client._received_messages[server] = np.array([0.0])  # noqa: SLF001

    local_x, z_next = algorithm._compute_local_update(client, server)

    np.testing.assert_allclose(local_x, np.array([-1.0]))
    np.testing.assert_allclose(z_next, np.array([-2.0]))


def test_fedlt_server_step_uses_server_cost_proximal_for_optional_global_regularizer() -> None:
    server_cost = ShiftProxCost(shift=0.5)
    clients = [Agent(0, ConstantGradientCost(0.0)), Agent(1, ConstantGradientCost(0.0))]
    server = Agent(2, server_cost)
    network = FedNetwork(clients=clients, server=server)
    algorithm = FedLT(iterations=1, step_size=0.1, num_local_epochs=1, rho=2.0)
    algorithm.initialize(network)
    server.aux_vars["z_by_client"][clients[0]] = np.array([1.0])
    server.aux_vars["z_by_client"][clients[1]] = np.array([3.0])

    y = algorithm._compute_server_y(network)

    np.testing.assert_allclose(y, np.array([2.5]))
    assert server_cost.proximal_rhos == [1.0]


def test_fedlt_server_step_supports_regularizer_server_cost() -> None:
    clients = [Agent(0, ConstantGradientCost(0.0)), Agent(1, ConstantGradientCost(0.0))]
    server = Agent(2, L1RegularizerCost(shape=(1,)))
    network = FedNetwork(clients=clients, server=server)
    algorithm = FedLT(iterations=1, step_size=0.1, num_local_epochs=1, rho=2.0)
    algorithm.initialize(network)
    server.aux_vars["z_by_client"][clients[0]] = np.array([3.0])
    server.aux_vars["z_by_client"][clients[1]] = np.array([1.0])

    y = algorithm._compute_server_y(network)

    np.testing.assert_allclose(y, np.array([1.0]))


def test_fedlt_empirical_cost_uses_existing_minibatch_gradient_default() -> None:
    cost = TrackingEmpiricalCost(n_samples=5, batch_size=2)
    client = Agent(0, cost)
    server = Agent(1, ZeroCost((1,)))
    algorithm = FedLT(iterations=1, step_size=1.0, num_local_epochs=3, rho=1.0)
    client.initialize(x=np.array([0.0]), aux_vars={"z": np.array([0.0])})
    server.initialize(x=np.array([0.0]))
    client._received_messages[server] = np.array([0.0])  # noqa: SLF001

    algorithm._compute_local_update(client, server)

    assert len(cost.gradient_indices) == 3
    assert all(len(indices) == 2 for indices in cost.gradient_indices)
    assert set().union(*(set(indices) for indices in cost.gradient_indices)) == set(range(cost.n_samples))


def test_fedlt_generic_cost_uses_full_gradient_call_default() -> None:
    cost = ConstantGradientCost(gradient_value=1.0)
    client = Agent(0, cost)
    server = Agent(1, ZeroCost((1,)))
    algorithm = FedLT(iterations=1, step_size=1.0, num_local_epochs=2, rho=1.0)
    client.initialize(x=np.array([0.0]), aux_vars={"z": np.array([0.0])})
    server.initialize(x=np.array([0.0]))
    client._received_messages[server] = np.array([0.0])  # noqa: SLF001

    algorithm._compute_local_update(client, server)

    assert cost.gradient_kwargs == [{}, {}]


def test_fedlt_supports_partial_participation_and_keeps_stale_server_z() -> None:
    network = _make_network(ConstantGradientCost(1.0), ConstantGradientCost(3.0))
    algorithm = FedLT(
        iterations=1,
        step_size=1.0,
        num_local_epochs=1,
        rho=1.0,
        selection_scheme=FirstClientSelection(),
    )

    algorithm.run(network)

    clients = network.clients()
    np.testing.assert_allclose(clients[0].x, np.array([-1.0]))
    np.testing.assert_allclose(clients[1].x, np.array([0.0]))
    np.testing.assert_allclose(network.server().aux_vars["z_by_client"][clients[0]], np.array([-2.0]))
    np.testing.assert_allclose(network.server().aux_vars["z_by_client"][clients[1]], np.array([0.0]))


def test_fedlt_smoke_with_network_noise_and_compression() -> None:
    network = FedNetwork(
        clients=[Agent(0, ConstantGradientCost(1.0)), Agent(1, ConstantGradientCost(2.0))],
        message_noise=GaussianNoise(0.0, 0.0),
        message_compression=Quantization(6),
    )
    algorithm = FedLT(iterations=2, step_size=0.1, num_local_epochs=1, rho=1.0)

    algorithm.run(network)

    assert np.isfinite(network.server().x).all()
