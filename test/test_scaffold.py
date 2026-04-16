from collections.abc import Sequence
from typing import Any

import numpy as np

from decent_bench.agents import Agent
from decent_bench.costs import Cost, ZeroCost
from decent_bench.distributed_algorithms import Scaffold
from decent_bench.networks import FedNetwork
from decent_bench.schemes import ClientSelectionScheme, DropScheme, NoDrops
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks


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


class DropOnCalls(DropScheme):
    def __init__(self, dropped_calls: set[int]):
        self._dropped_calls = dropped_calls
        self._call_count = 0

    def should_drop(self) -> bool:
        self._call_count += 1
        return self._call_count in self._dropped_calls


class ScheduledSelection(ClientSelectionScheme):
    def __init__(self, selected_ids_by_round: dict[int, list[int]]):
        self._selected_ids_by_round = selected_ids_by_round

    def select(self, clients: Sequence[Agent], iteration: int) -> list[Agent]:
        selected_ids = set(self._selected_ids_by_round.get(iteration, []))
        return [client for client in clients if client.id in selected_ids]


def _run_scaffold_local_update(
    cost: Cost,
    *,
    step_size: float = 1.0,
    num_local_epochs: int = 1,
    client_control: float = 0.0,
    server_control: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    algorithm = Scaffold(iterations=1, step_size=step_size, num_local_epochs=num_local_epochs)
    client = Agent(0, cost)
    server = Agent(1, ZeroCost(cost.shape))
    client.initialize(
        x=np.zeros(cost.shape, dtype=float),
        aux_vars={
            "c_i": np.array([client_control], dtype=float),
            "c": np.array([server_control], dtype=float),
        },
    )
    server.initialize(x=np.zeros(cost.shape, dtype=float))
    local_x, control_delta = algorithm._compute_local_update(client, server)
    return local_x, client.aux_vars["c_i"], control_delta


def test_control_variate_correction_changes_the_local_step() -> None:
    cost = TrackingCost(gradient_value=1.0)

    updated, client_control, control_delta = _run_scaffold_local_update(
        cost,
        num_local_epochs=2,
        client_control=1.5,
        server_control=0.5,
    )

    np.testing.assert_allclose(updated, np.array([0.0]))
    np.testing.assert_allclose(client_control, np.array([1.0]))
    np.testing.assert_allclose(control_delta, np.array([-0.5]))


def test_scaffold_persists_control_variates_across_rounds() -> None:
    algorithm = Scaffold(
        iterations=2,
        step_size=1.0,
        num_local_epochs=1,
        server_step_size=1.0,
    )
    clients = [Agent(0, TrackingCost(1.0)), Agent(1, TrackingCost(3.0))]
    network = FedNetwork(clients=clients)
    algorithm.initialize(network)

    network._step(0)  # noqa: SLF001
    algorithm.step(network, 0)

    np.testing.assert_allclose(network.server().x, np.array([-2.0]))
    np.testing.assert_allclose(network.server().aux_vars["c"], np.array([2.0]))
    np.testing.assert_allclose(clients[0].aux_vars["c_i"], np.array([1.0]))
    np.testing.assert_allclose(clients[1].aux_vars["c_i"], np.array([3.0]))

    network._step(1)  # noqa: SLF001
    algorithm.step(network, 1)

    np.testing.assert_allclose(network.server().x, np.array([-4.0]))
    np.testing.assert_allclose(network.server().aux_vars["c"], np.array([2.0]))
    np.testing.assert_allclose(clients[0].aux_vars["c_i"], np.array([1.0]))
    np.testing.assert_allclose(clients[1].aux_vars["c_i"], np.array([3.0]))


def test_scaffold_uses_uniform_aggregation() -> None:
    algorithm = Scaffold(iterations=1, step_size=1.0, num_local_epochs=1)
    network = FedNetwork(clients=[Agent(0, TrackingCost(1.0)), Agent(1, TrackingCost(3.0))])

    algorithm.initialize(network)
    network._step(0)  # noqa: SLF001
    algorithm.step(network, 0)

    np.testing.assert_allclose(network.server().x, np.array([-2.0]))
    np.testing.assert_allclose(network.server().aux_vars["c"], np.array([2.0]))


def test_server_step_size_scales_only_the_server_model_update() -> None:
    full_step_algorithm = Scaffold(
        iterations=1,
        step_size=1.0,
        num_local_epochs=1,
        server_step_size=1.0,
    )
    damped_step_algorithm = Scaffold(
        iterations=1,
        step_size=1.0,
        num_local_epochs=1,
        server_step_size=0.25,
    )
    full_step_network = FedNetwork(clients=[Agent(0, TrackingCost(1.0)), Agent(1, TrackingCost(3.0))])
    damped_step_network = FedNetwork(clients=[Agent(0, TrackingCost(1.0)), Agent(1, TrackingCost(3.0))])

    full_step_algorithm.initialize(full_step_network)
    damped_step_algorithm.initialize(damped_step_network)
    full_step_network._step(0)  # noqa: SLF001
    damped_step_network._step(0)  # noqa: SLF001
    full_step_algorithm.step(full_step_network, 0)
    damped_step_algorithm.step(damped_step_network, 0)

    np.testing.assert_allclose(full_step_network.server().x, np.array([-2.0]))
    np.testing.assert_allclose(damped_step_network.server().x, np.array([-0.5]))
    np.testing.assert_allclose(damped_step_network.server().x, 0.25 * full_step_network.server().x)
    np.testing.assert_allclose(full_step_network.server().aux_vars["c"], np.array([2.0]))
    np.testing.assert_allclose(damped_step_network.server().aux_vars["c"], np.array([2.0]))


def test_scaffold_aggregation_uses_only_received_updates_for_model_and_control_deltas() -> None:
    algorithm = Scaffold(iterations=1, step_size=1.0, num_local_epochs=1)
    clients = [Agent(0, TrackingCost(1.0)), Agent(1, TrackingCost(2.0))]
    network = FedNetwork(clients=clients)
    algorithm.initialize(network)
    network.server().x = np.array([10.0])
    network.server().aux_vars["c"] = np.array([5.0])
    clients[0].aux_vars["delta_c"] = np.array([4.0])
    clients[1].aux_vars["delta_c"] = np.array([100.0])

    network.send(sender=clients[0], receiver=network.server(), msg=np.array([12.0]))

    algorithm.aggregate(network, clients)

    np.testing.assert_allclose(network.server().x, np.array([12.0]))
    np.testing.assert_allclose(network.server().aux_vars["c"], np.array([7.0]))


def test_scaffold_partial_participation_persists_control_variates_across_rounds() -> None:
    algorithm = Scaffold(
        iterations=3,
        step_size=1.0,
        num_local_epochs=1,
        server_step_size=1.0,
        selection_scheme=ScheduledSelection({0: [0, 1], 1: [1], 2: [2]}),
    )
    clients = [Agent(0, TrackingCost(1.0)), Agent(1, TrackingCost(2.0)), Agent(2, TrackingCost(3.0))]
    network = FedNetwork(clients=clients)
    algorithm.initialize(network)

    network._step(0)  # noqa: SLF001
    algorithm.step(network, 0)

    np.testing.assert_allclose(network.server().x, np.array([-1.5]))
    np.testing.assert_allclose(network.server().aux_vars["c"], np.array([1.0]))
    np.testing.assert_allclose(clients[0].x, np.array([-1.0]))
    np.testing.assert_allclose(clients[1].x, np.array([-2.0]))
    np.testing.assert_allclose(clients[2].x, np.array([0.0]))
    np.testing.assert_allclose(clients[0].aux_vars["c_i"], np.array([1.0]))
    np.testing.assert_allclose(clients[1].aux_vars["c_i"], np.array([2.0]))
    np.testing.assert_allclose(clients[2].aux_vars["c_i"], np.array([0.0]))

    network._step(1)  # noqa: SLF001
    algorithm.step(network, 1)

    np.testing.assert_allclose(network.server().x, np.array([-2.5]))
    np.testing.assert_allclose(network.server().aux_vars["c"], np.array([1.0]))
    np.testing.assert_allclose(clients[0].x, np.array([-1.0]))
    np.testing.assert_allclose(clients[1].x, np.array([-2.5]))
    np.testing.assert_allclose(clients[2].x, np.array([0.0]))
    np.testing.assert_allclose(clients[0].aux_vars["c_i"], np.array([1.0]))
    np.testing.assert_allclose(clients[1].aux_vars["c_i"], np.array([2.0]))
    np.testing.assert_allclose(clients[2].aux_vars["c_i"], np.array([0.0]))
    np.testing.assert_allclose(clients[0].aux_vars["c"], np.array([0.0]))
    np.testing.assert_allclose(clients[1].aux_vars["c"], np.array([1.0]))
    np.testing.assert_allclose(clients[2].aux_vars["c"], np.array([0.0]))

    network._step(2)  # noqa: SLF001
    algorithm.step(network, 2)

    np.testing.assert_allclose(network.server().x, np.array([-6.5]))
    np.testing.assert_allclose(network.server().aux_vars["c"], np.array([2.0]))
    np.testing.assert_allclose(clients[0].x, np.array([-1.0]))
    np.testing.assert_allclose(clients[1].x, np.array([-2.5]))
    np.testing.assert_allclose(clients[2].x, np.array([-6.5]))
    np.testing.assert_allclose(clients[0].aux_vars["c_i"], np.array([1.0]))
    np.testing.assert_allclose(clients[1].aux_vars["c_i"], np.array([2.0]))
    np.testing.assert_allclose(clients[2].aux_vars["c_i"], np.array([3.0]))
    np.testing.assert_allclose(clients[0].aux_vars["c"], np.array([0.0]))
    np.testing.assert_allclose(clients[1].aux_vars["c"], np.array([1.0]))
    np.testing.assert_allclose(clients[2].aux_vars["c"], np.array([1.0]))


def test_scaffold_keeps_cached_server_control_when_broadcast_is_dropped() -> None:
    algorithm = Scaffold(iterations=2, step_size=1.0, num_local_epochs=1)
    client = Agent(0, TrackingCost(gradient_value=0.0))
    server = Agent(1, ZeroCost((1,)))
    network = FedNetwork(
        clients=[client],
        server=server,
        message_drop={server: DropOnCalls({2}), client: NoDrops()},
    )
    algorithm.initialize(network)

    server.aux_vars["c"] = np.array([2.0])
    network._step(0)  # noqa: SLF001
    algorithm.step(network, 0)

    np.testing.assert_allclose(client.aux_vars["c"], np.array([2.0]))
    np.testing.assert_allclose(network.server().x, np.array([-2.0]))

    server.aux_vars["c"] = np.array([5.0])
    network._step(1)  # noqa: SLF001
    algorithm.step(network, 1)

    np.testing.assert_allclose(client.aux_vars["c"], np.array([2.0]))
    np.testing.assert_allclose(network.server().x, np.array([-4.0]))
