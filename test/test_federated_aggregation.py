from typing import Any

import numpy as np

from decent_bench.agents import Agent
from decent_bench.costs import Cost
from decent_bench.distributed_algorithms import FedAvg
from decent_bench.networks import FedNetwork
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks


class TrackingCost(Cost):
    def __init__(self, gradient_value: float = 1.0):
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
        del x, kwargs
        return self._gradient.copy()

    def hessian(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        del x, kwargs
        return np.zeros((1, 1), dtype=float)

    def proximal(self, x: np.ndarray, rho: float, **kwargs: Any) -> np.ndarray:
        del rho, kwargs
        return x


def _make_fed_network(*costs: Cost) -> tuple[FedNetwork, list[Agent]]:
    clients = [Agent(i, cost) for i, cost in enumerate(costs)]
    network = FedNetwork(clients=clients)
    for client in clients:
        client.initialize(x=np.zeros(client.cost.shape, dtype=float))
    network.server().initialize(x=np.zeros(clients[0].cost.shape, dtype=float))
    return network, clients


def test_aggregation_uses_only_received_client_updates() -> None:
    algorithm = FedAvg(iterations=1, step_size=1.0)
    network, clients = _make_fed_network(TrackingCost(1.0), TrackingCost(2.0))

    network.send(sender=clients[0], receiver=network.server(), msg=np.array([3.0]))

    algorithm.aggregate(network, clients, client_weights={clients[0]: 1.0, clients[1]: 10.0})

    np.testing.assert_allclose(network.server().x, np.array([3.0]))


def test_aggregation_keeps_server_state_when_no_updates_are_received() -> None:
    algorithm = FedAvg(iterations=1, step_size=1.0)
    network, clients = _make_fed_network(TrackingCost(1.0), TrackingCost(2.0))
    network.server().x = np.array([7.0])

    algorithm.aggregate(network, clients)

    np.testing.assert_allclose(network.server().x, np.array([7.0]))
