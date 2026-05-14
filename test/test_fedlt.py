from collections.abc import Sequence
import math
from typing import Any

import numpy as np
import pytest

from decent_bench.agents import Agent
from decent_bench.algorithms.federated import FedLT
from decent_bench.costs import Cost, EmpiricalRiskCost, L1RegularizerCost, QuadraticCost, ZeroCost
from decent_bench.networks import FedNetwork
from decent_bench.schemes import ClientSelectionScheme, DropScheme, GaussianNoise, NoDrops, Quantization
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


class DropOnCalls(DropScheme):
    def __init__(self, dropped_calls: set[int]):
        self._dropped_calls = dropped_calls
        self._call_count = 0

    def should_drop(self) -> bool:
        self._call_count += 1
        return self._call_count in self._dropped_calls


def _make_network(*costs: Cost) -> FedNetwork:
    return FedNetwork(clients=[Agent(cost) for cost in costs])


@pytest.mark.parametrize("local_solver", ["gd", "nesterov"])
def test_fedlt_runs_with_default_and_nesterov_local_updates(local_solver: str) -> None:
    network = _make_network(
        QuadraticCost(A=np.array([[1.0]]), b=np.array([-1.0])),
        QuadraticCost(A=np.array([[2.0]]), b=np.array([1.0])),
    )
    algorithm = FedLT(iterations=3, step_size=0.2, num_local_epochs=2, rho=1.0, local_solver=local_solver)

    algorithm.run(network)

    assert np.isfinite(network.server().x).all()
    assert all(np.isfinite(client.x).all() for client in network.clients())


def test_fedlt_accepts_adam_local_solver() -> None:
    network = _make_network(
        QuadraticCost(A=np.array([[1.0]]), b=np.array([-1.0])),
        QuadraticCost(A=np.array([[2.0]]), b=np.array([1.0])),
    )
    algorithm = FedLT(iterations=3, step_size=0.2, num_local_epochs=2, rho=1.0, local_solver="adam")

    algorithm.run(network)

    assert np.isfinite(network.server().x).all()
    assert all(np.isfinite(client.x).all() for client in network.clients())


def test_fedlt_sets_default_solver_args() -> None:
    nesterov = FedLT(iterations=1, local_solver="nesterov")
    adam = FedLT(iterations=1, local_solver="adam")

    assert nesterov.solver_args == {"momentum": 0.9}
    assert adam.solver_args == {"beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8}


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"rho": 0.0}, "`rho` must be positive"),
        ({"num_local_epochs": 0}, "`num_local_epochs` must be positive"),
        ({"step_size": 0.0}, "`step_size` must be positive"),
        ({"local_solver": "bad"}, "`local_solver` must be one of"),
        (
            {"local_solver": "adam", "solver_args": {"beta1": -0.1}},
            "`solver_args\\['beta1'\\]` must satisfy 0 <= beta1 < 1",
        ),
        (
            {"local_solver": "adam", "solver_args": {"beta1": 1.0}},
            "`solver_args\\['beta1'\\]` must satisfy 0 <= beta1 < 1",
        ),
        (
            {"local_solver": "adam", "solver_args": {"beta2": -0.1}},
            "`solver_args\\['beta2'\\]` must satisfy 0 <= beta2 < 1",
        ),
        (
            {"local_solver": "adam", "solver_args": {"beta2": 1.0}},
            "`solver_args\\['beta2'\\]` must satisfy 0 <= beta2 < 1",
        ),
        ({"local_solver": "adam", "solver_args": {"epsilon": 0.0}}, "`solver_args\\['epsilon'\\]` must be positive"),
        (
            {"local_solver": "nesterov", "solver_args": {"momentum": -0.1}},
            "`solver_args\\['momentum'\\]` must satisfy 0 <= momentum < 1",
        ),
        (
            {"local_solver": "nesterov", "solver_args": {"momentum": 1.0}},
            "`solver_args\\['momentum'\\]` must satisfy 0 <= momentum < 1",
        ),
        ({"solver_args": {"momentum": 0.5}}, "Unsupported solver_args for local_solver='gd': momentum"),
        (
            {"local_solver": "adam", "solver_args": {"momentum": 0.5}},
            "Unsupported solver_args for local_solver='adam': momentum",
        ),
    ],
)
def test_fedlt_rejects_invalid_parameters(kwargs: dict[str, float | int | str | dict[str, float]], error: str) -> None:
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


def test_fedlt_nesterov_update_uses_step_size_and_momentum() -> None:
    client = Agent(ConstantGradientCost(gradient_value=2.0))
    server = Agent(ZeroCost((1,)))
    algorithm = FedLT(
        iterations=1,
        step_size=0.25,
        num_local_epochs=2,
        rho=1.0,
        local_solver="nesterov",
        solver_args={"momentum": 0.5},
    )
    client.initialize(x=np.array([1.0]), aux_vars={"z": np.array([0.0])})
    server.initialize(x=np.array([0.0]))
    client._received_messages[server] = np.array([0.0])  # noqa: SLF001

    local_x, z_next = algorithm._compute_local_update(client, server)

    np.testing.assert_allclose(local_x, np.array([-1.015625]))
    np.testing.assert_allclose(z_next, np.array([-2.03125]))


def test_fedlt_nesterov_default_momentum_is_used() -> None:
    client = Agent(ConstantGradientCost(gradient_value=2.0))
    server = Agent(ZeroCost((1,)))
    algorithm = FedLT(
        iterations=1,
        step_size=0.25,
        num_local_epochs=2,
        rho=1.0,
        local_solver="nesterov",
    )
    client.initialize(x=np.array([1.0]), aux_vars={"z": np.array([0.0])})
    server.initialize(x=np.array([0.0]))
    client._received_messages[server] = np.array([0.0])  # noqa: SLF001

    local_x, z_next = algorithm._compute_local_update(client, server)

    np.testing.assert_allclose(local_x, np.array([-1.780625]))
    np.testing.assert_allclose(z_next, np.array([-3.56125]))


def test_fedlt_local_gradient_step_uses_penalty_term() -> None:
    client = Agent(ConstantGradientCost(gradient_value=1.0))
    server = Agent(ZeroCost((1,)))
    algorithm = FedLT(iterations=1, step_size=1.0, num_local_epochs=2, rho=1.0)
    client.initialize(x=np.array([0.0]), aux_vars={"z": np.array([0.0])})
    server.initialize(x=np.array([0.0]))
    client._received_messages[server] = np.array([0.0])  # noqa: SLF001

    local_x, z_next = algorithm._compute_local_update(client, server)

    np.testing.assert_allclose(local_x, np.array([-1.0]))
    np.testing.assert_allclose(z_next, np.array([-2.0]))


def test_fedlt_adam_one_step_matches_formula() -> None:
    client = Agent(ConstantGradientCost(gradient_value=2.0))
    server = Agent(ZeroCost((1,)))
    algorithm = FedLT(iterations=1, step_size=0.5, num_local_epochs=1, rho=1.0, local_solver="adam")
    client.initialize(x=np.array([0.0]), aux_vars={"z": np.array([0.0])})
    server.initialize(x=np.array([0.0]))
    client._received_messages[server] = np.array([0.0])  # noqa: SLF001

    local_x, z_next = algorithm._compute_local_update(client, server)

    expected_x = np.array([-0.5 * 2.0 / (math.sqrt(4.0) + 1e-8)])
    np.testing.assert_allclose(local_x, expected_x)
    np.testing.assert_allclose(z_next, 2 * expected_x)


def _manual_adam_steps(
    *,
    x0: float,
    step_size: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    num_steps: int,
) -> float:
    x = x0
    m = 0.0
    v = 0.0
    for step in range(1, num_steps + 1):
        grad = 2 * x
        m = (beta1 * m) + ((1 - beta1) * grad)
        v = (beta2 * v) + ((1 - beta2) * (grad * grad))
        m_hat = m / (1 - (beta1**step))
        v_hat = v / (1 - (beta2**step))
        x -= step_size * m_hat / (math.sqrt(v_hat) + epsilon)
    return x


def test_fedlt_adam_multi_step_matches_formula_on_quadratic() -> None:
    client = Agent(QuadraticCost(A=np.array([[1.0]]), b=np.array([0.0])))
    server = Agent(ZeroCost((1,)))
    algorithm = FedLT(
        iterations=1,
        step_size=0.1,
        num_local_epochs=3,
        rho=1.0,
        local_solver="adam",
        solver_args={"beta1": 0.8, "beta2": 0.9, "epsilon": 1e-6},
    )
    client.initialize(x=np.array([1.0]), aux_vars={"z": np.array([0.0])})
    server.initialize(x=np.array([0.0]))
    client._received_messages[server] = np.array([0.0])  # noqa: SLF001

    local_x, z_next = algorithm._compute_local_update(client, server)

    expected_x = _manual_adam_steps(
        x0=1.0,
        step_size=algorithm.step_size,
        beta1=algorithm.solver_args["beta1"],
        beta2=algorithm.solver_args["beta2"],
        epsilon=algorithm.solver_args["epsilon"],
        num_steps=algorithm.num_local_epochs,
    )
    np.testing.assert_allclose(local_x, np.array([expected_x]))
    np.testing.assert_allclose(z_next, np.array([2 * expected_x]))


def test_fedlt_adam_moments_reset_each_local_solve() -> None:
    client = Agent(QuadraticCost(A=np.array([[1.0]]), b=np.array([0.0])))
    server = Agent(ZeroCost((1,)))
    algorithm = FedLT(iterations=1, step_size=0.1, num_local_epochs=2, rho=1.0, local_solver="adam")
    server.initialize(x=np.array([0.0]))

    def run_local_solve() -> tuple[np.ndarray, np.ndarray]:
        client.initialize(x=np.array([1.0]), aux_vars={"z": np.array([0.0])})
        client._received_messages[server] = np.array([0.0])  # noqa: SLF001
        return algorithm._compute_local_update(client, server)

    local_x_1, z_next_1 = run_local_solve()
    local_x_2, z_next_2 = run_local_solve()

    np.testing.assert_allclose(local_x_1, local_x_2)
    np.testing.assert_allclose(z_next_1, z_next_2)
    assert set(client.aux_vars) == {"z"}


def test_fedlt_server_step_uses_server_cost_proximal_for_optional_global_regularizer() -> None:
    server_cost = ShiftProxCost(shift=0.5)
    clients = [Agent(ConstantGradientCost(0.0)), Agent(ConstantGradientCost(0.0))]
    server = Agent(server_cost)
    network = FedNetwork(clients=clients, server=server)
    algorithm = FedLT(iterations=1, step_size=0.1, num_local_epochs=1, rho=2.0)
    algorithm.initialize(network)
    server.aux_vars["z_by_client"][clients[0]] = np.array([1.0])
    server.aux_vars["z_by_client"][clients[1]] = np.array([3.0])

    y = algorithm._compute_server_y(network)

    np.testing.assert_allclose(y, np.array([2.5]))
    assert server_cost.proximal_rhos == [1.0]


def test_fedlt_server_step_supports_regularizer_server_cost() -> None:
    clients = [Agent(ConstantGradientCost(0.0)), Agent(ConstantGradientCost(0.0))]
    server = Agent(L1RegularizerCost(shape=(1,)))
    network = FedNetwork(clients=clients, server=server)
    algorithm = FedLT(iterations=1, step_size=0.1, num_local_epochs=1, rho=2.0)
    algorithm.initialize(network)
    server.aux_vars["z_by_client"][clients[0]] = np.array([3.0])
    server.aux_vars["z_by_client"][clients[1]] = np.array([1.0])

    y = algorithm._compute_server_y(network)

    np.testing.assert_allclose(y, np.array([1.0]))


def test_fedlt_empirical_cost_uses_existing_minibatch_gradient_default() -> None:
    cost = TrackingEmpiricalCost(n_samples=5, batch_size=2)
    client = Agent(cost)
    server = Agent(ZeroCost((1,)))
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
    client = Agent(cost)
    server = Agent(ZeroCost((1,)))
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


def test_fedlt_keeps_stale_server_z_when_client_upload_is_dropped() -> None:
    clients = [Agent(ConstantGradientCost(1.0)), Agent(ConstantGradientCost(3.0))]
    network = FedNetwork(
        clients=clients,
        message_drop={clients[0]: DropOnCalls({1}), clients[1]: NoDrops()},
    )
    algorithm = FedLT(iterations=1, step_size=1.0, num_local_epochs=1, rho=1.0)
    algorithm.initialize(network)

    network._step(0)  # noqa: SLF001
    algorithm.step(network, 0)

    np.testing.assert_allclose(clients[0].x, np.array([-1.0]))
    np.testing.assert_allclose(clients[0].aux_vars["z"], np.array([-2.0]))
    np.testing.assert_allclose(clients[1].x, np.array([-3.0]))
    np.testing.assert_allclose(clients[1].aux_vars["z"], np.array([-6.0]))
    np.testing.assert_allclose(network.server().aux_vars["z_by_client"][clients[0]], np.array([0.0]))
    np.testing.assert_allclose(network.server().aux_vars["z_by_client"][clients[1]], np.array([-6.0]))


def test_fedlt_smoke_with_network_noise_and_compression() -> None:
    network = FedNetwork(
        clients=[Agent(ConstantGradientCost(1.0)), Agent(ConstantGradientCost(2.0))],
        message_noise=GaussianNoise(0.0, 0.0),
        message_compression=Quantization(quantization_step=1e-2),
    )
    algorithm = FedLT(iterations=2, step_size=0.1, num_local_epochs=1, rho=1.0)

    algorithm.run(network)

    assert np.isfinite(network.server().x).all()
