from typing import Any

import numpy as np
import pytest

from decent_bench.agents import Agent
from decent_bench.costs import Cost
from decent_bench.distributed_algorithms import FedAvg, FedNova
from decent_bench.networks import FedNetwork
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks


class TrackingCost(Cost):
    def __init__(self, gradient_value: float = 1.0, *, n_samples: int = 1):
        self._gradient = np.array([gradient_value], dtype=float)
        self.n_samples = n_samples

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
        del x, kwargs
        return self._gradient.copy()

    def hessian(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        del x, kwargs
        return np.zeros((1, 1), dtype=float)

    def proximal(self, x: np.ndarray, rho: float, **kwargs: Any) -> np.ndarray:
        del rho, kwargs
        return x


def _make_fed_network(*client_specs: float | tuple[float, int]) -> FedNetwork:
    clients = []
    for i, client_spec in enumerate(client_specs):
        if isinstance(client_spec, tuple):
            gradient_value, n_samples = client_spec
        else:
            gradient_value, n_samples = client_spec, 1
        clients.append(Agent(i, TrackingCost(gradient_value, n_samples=n_samples)))
    return FedNetwork(clients=clients)


def test_fednova_one_round_smoke_matches_fedavg_for_homogeneous_local_steps() -> None:
    network = _make_fed_network(1.0, 3.0)
    algorithm = FedNova(iterations=1, step_size=1.0, local_steps=2)

    algorithm.initialize(network)
    network._step(0)  # noqa: SLF001
    algorithm.step(network, 0)

    np.testing.assert_allclose(network.server().x, np.array([-4.0]))


def test_fednova_supports_heterogeneous_local_steps() -> None:
    network = _make_fed_network(1.0, 3.0)
    clients = network.clients()
    algorithm = FedNova(iterations=1, step_size=1.0, local_steps={clients[0]: 1, clients[1]: 3})

    algorithm.initialize(network)
    network._step(0)  # noqa: SLF001
    algorithm.step(network, 0)

    np.testing.assert_allclose(network.server().x, np.array([-4.0]))


def test_fednova_uses_data_proportional_client_weights() -> None:
    network = _make_fed_network((1.0, 1), (3.0, 3))
    algorithm = FedNova(iterations=1, step_size=1.0, local_steps=1)

    algorithm.initialize(network)
    network._step(0)  # noqa: SLF001
    algorithm.step(network, 0)

    np.testing.assert_allclose(network.server().x, np.array([-2.5]))


def test_fednova_uploads_cumulative_gradient_normalizer_and_client_weight() -> None:
    network = _make_fed_network((2.0, 1), (4.0, 3))
    clients = network.clients()
    algorithm = FedNova(iterations=1, step_size=1.0, local_steps={clients[0]: 2, clients[1]: 1})

    algorithm.initialize(network)
    network._step(0)  # noqa: SLF001
    selected_clients = network.clients()
    algorithm.server_broadcast(network, selected_clients)
    participating_clients = algorithm._clients_with_server_broadcast(network, selected_clients)
    algorithm._clear_buffered_server_messages(network, participating_clients)
    algorithm._run_local_updates(network, participating_clients)

    client_0_payload = algorithm._unpack_local_update_payload(network.server().messages[clients[0]], (1,))  # noqa: SLF001
    client_1_payload = algorithm._unpack_local_update_payload(network.server().messages[clients[1]], (1,))  # noqa: SLF001

    np.testing.assert_allclose(client_0_payload[0], np.array([4.0]))
    np.testing.assert_allclose(client_1_payload[0], np.array([4.0]))
    assert client_0_payload[1] == 2.0
    assert client_1_payload[1] == 1.0
    assert client_0_payload[2] == 0.25
    assert client_1_payload[2] == 0.75


def test_fednova_differs_from_fedavg_when_local_steps_are_heterogeneous() -> None:
    fednova_network = _make_fed_network(1.0, 3.0)
    fedavg_network = _make_fed_network(1.0, 3.0)
    fednova_clients = fednova_network.clients()
    fednova = FedNova(
        iterations=1,
        step_size=1.0,
        local_steps={fednova_clients[0]: 1, fednova_clients[1]: 3},
    )
    fedavg = FedAvg(iterations=1, step_size=1.0, num_local_epochs=1)

    fednova.initialize(fednova_network)
    fedavg.initialize(fedavg_network)

    fednova_network._step(0)  # noqa: SLF001
    fedavg_network._step(0)  # noqa: SLF001

    fednova.step(fednova_network, 0)
    fedavg.step(fedavg_network, 0)

    np.testing.assert_allclose(fednova_network.server().x, np.array([-4.0]))
    np.testing.assert_allclose(fedavg_network.server().x, np.array([-2.0]))
    assert not np.allclose(fednova_network.server().x, fedavg_network.server().x)


def test_fednova_differs_from_uniform_weighting_when_client_sizes_differ() -> None:
    fednova_network = _make_fed_network((1.0, 1), (3.0, 3))
    fedavg_network = _make_fed_network((1.0, 1), (3.0, 3))
    fednova_clients = fednova_network.clients()
    fednova = FedNova(
        iterations=1,
        step_size=1.0,
        local_steps={fednova_clients[0]: 1, fednova_clients[1]: 3},
    )
    fedavg = FedAvg(iterations=1, step_size=1.0, num_local_epochs=1)

    fednova.initialize(fednova_network)
    fedavg.initialize(fedavg_network)

    fednova_network._step(0)  # noqa: SLF001
    fedavg_network._step(0)  # noqa: SLF001

    fednova.step(fednova_network, 0)
    fedavg.step(fedavg_network, 0)

    np.testing.assert_allclose(fednova_network.server().x, np.array([-6.25]))
    np.testing.assert_allclose(fedavg_network.server().x, np.array([-2.0]))
    assert not np.allclose(fednova_network.server().x, fedavg_network.server().x)


def test_fednova_rejects_sequence_local_steps() -> None:
    with pytest.raises(TypeError, match="`local_steps` must be an int or a mapping from Agent"):
        FedNova(iterations=1, step_size=1.0, local_steps=[1, 3])


def test_fednova_rejects_local_step_mappings_that_do_not_match_network_clients() -> None:
    network = _make_fed_network(1.0, 3.0)
    other_network = _make_fed_network(2.0)
    algorithm = FedNova(
        iterations=1,
        step_size=1.0,
        local_steps={network.clients()[0]: 1, other_network.clients()[0]: 3},
    )

    with pytest.raises(ValueError, match="`local_steps` mapping must match the network clients exactly"):
        algorithm.initialize(network)
