from typing import Any

import numpy as np
import pytest

from decent_bench.agents import Agent
from decent_bench.costs import Cost
from decent_bench.distributed_algorithms import FedAvg, FedProx
from decent_bench.networks import FedNetwork
from decent_bench.schemes import DropScheme, NoDrops
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks


class TrackingCost(Cost):
    def __init__(self, gradient_value: float = 1.0, *, n_samples: int | None = None):
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


class DropOnCalls(DropScheme):
    def __init__(self, dropped_calls: set[int]):
        self._dropped_calls = dropped_calls
        self._call_count = 0

    def should_drop(self) -> bool:
        self._call_count += 1
        return self._call_count in self._dropped_calls


def _make_fed_network(*costs: Cost) -> tuple[FedNetwork, list[Agent]]:
    clients = [Agent(i, cost) for i, cost in enumerate(costs)]
    network = FedNetwork(clients=clients)
    for client in clients:
        client.initialize(x=np.zeros(client.cost.shape, dtype=float))
    network.server().initialize(x=np.zeros(clients[0].cost.shape, dtype=float))
    return network, clients


@pytest.mark.parametrize(
    ("algorithm_cls", "kwargs"),
    [
        pytest.param(FedAvg, {"iterations": 1, "step_size": 1.0}, id="fedavg"),
        pytest.param(FedProx, {"iterations": 1, "step_size": 1.0}, id="fedprox"),
    ],
)
def test_aggregation_uses_only_received_client_updates(
    algorithm_cls: type[FedAvg] | type[FedProx], kwargs: dict[str, float | int]
) -> None:
    algorithm = algorithm_cls(**kwargs)
    network, clients = _make_fed_network(TrackingCost(1.0), TrackingCost(2.0))

    network.send(sender=clients[0], receiver=network.server(), msg=np.array([3.0]))

    algorithm.aggregate(network, clients)

    np.testing.assert_allclose(network.server().x, np.array([3.0]))


@pytest.mark.parametrize(
    ("algorithm_cls", "kwargs"),
    [
        pytest.param(FedAvg, {"iterations": 1, "step_size": 1.0}, id="fedavg"),
        pytest.param(FedProx, {"iterations": 1, "step_size": 1.0}, id="fedprox"),
    ],
)
def test_default_aggregation_is_uniform(
    algorithm_cls: type[FedAvg] | type[FedProx], kwargs: dict[str, float | int]
) -> None:
    algorithm = algorithm_cls(**kwargs)
    network, clients = _make_fed_network(
        TrackingCost(gradient_value=1.0, n_samples=1),
        TrackingCost(gradient_value=3.0, n_samples=3),
    )

    network.send(sender=clients[0], receiver=network.server(), msg=np.array([-1.0]))
    network.send(sender=clients[1], receiver=network.server(), msg=np.array([-3.0]))

    algorithm.aggregate(network, clients)

    np.testing.assert_allclose(network.server().x, np.array([-2.0]))


@pytest.mark.parametrize(
    ("algorithm_cls", "kwargs"),
    [
        pytest.param(FedAvg, {"iterations": 1, "step_size": 1.0}, id="fedavg"),
        pytest.param(FedProx, {"iterations": 1, "step_size": 1.0}, id="fedprox"),
    ],
)
def test_aggregation_keeps_server_state_when_no_updates_are_received(
    algorithm_cls: type[FedAvg] | type[FedProx], kwargs: dict[str, float | int]
) -> None:
    algorithm = algorithm_cls(**kwargs)
    network, clients = _make_fed_network(TrackingCost(1.0), TrackingCost(2.0))
    network.server().x = np.array([7.0])

    algorithm.aggregate(network, clients)

    np.testing.assert_allclose(network.server().x, np.array([7.0]))


@pytest.mark.parametrize(
    ("algorithm_cls", "kwargs"),
    [
        pytest.param(FedAvg, {"iterations": 2, "step_size": 1.0}, id="fedavg"),
        pytest.param(FedProx, {"iterations": 2, "step_size": 1.0}, id="fedprox"),
    ],
)
def test_clients_without_server_broadcast_do_not_participate(
    algorithm_cls: type[FedAvg] | type[FedProx], kwargs: dict[str, float | int]
) -> None:
    client = Agent(0, TrackingCost(1.0))
    server = Agent(1, TrackingCost(0.0))
    network = FedNetwork(
        clients=[client],
        server=server,
        message_drop={server: DropOnCalls({2}), client: NoDrops()},
    )
    algorithm = algorithm_cls(**kwargs)
    algorithm.initialize(network)

    network._step(0)  # noqa: SLF001
    algorithm.step(network, 0)
    np.testing.assert_allclose(network.server().x, np.array([-1.0]))

    network._step(1)  # noqa: SLF001
    algorithm.step(network, 1)

    np.testing.assert_allclose(network.server().x, np.array([-1.0]))


@pytest.mark.parametrize(
    ("algorithm_cls", "kwargs"),
    [
        pytest.param(FedAvg, {"iterations": 2, "step_size": 1.0}, id="fedavg"),
        pytest.param(FedProx, {"iterations": 2, "step_size": 1.0}, id="fedprox"),
    ],
)
def test_buffered_stale_client_messages_are_not_aggregated(
    algorithm_cls: type[FedAvg] | type[FedProx], kwargs: dict[str, float | int]
) -> None:
    clients = [Agent(0, TrackingCost(1.0)), Agent(1, TrackingCost(3.0))]
    server = Agent(2, TrackingCost(0.0))
    network = FedNetwork(
        clients=clients,
        server=server,
        buffer_messages=True,
        message_drop={server: NoDrops(), clients[0]: DropOnCalls({2}), clients[1]: NoDrops()},
    )
    algorithm = algorithm_cls(**kwargs)
    algorithm.initialize(network)

    network._step(0)  # noqa: SLF001
    algorithm.step(network, 0)
    np.testing.assert_allclose(network.server().x, np.array([-2.0]))

    network._step(1)  # noqa: SLF001
    algorithm.step(network, 1)

    np.testing.assert_allclose(network.server().x, np.array([-5.0]))
