from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from decent_bench.agents import Agent
from decent_bench.algorithms.federated import FedDyn
from decent_bench.costs import Cost, ZeroCost
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


class FirstClientSelection(ClientSelectionScheme):
    def select(self, clients: Sequence[Agent], iteration: int) -> list[Agent]:
        del iteration
        return [clients[0]]


def test_feddyn_initializes_server_and_client_dynamic_states() -> None:
    clients = [Agent(TrackingCost(1.0)), Agent(TrackingCost(2.0))]
    network = FedNetwork(clients=clients)
    algorithm = FedDyn(iterations=1, x0=np.array([2.0]))

    algorithm.initialize(network)

    np.testing.assert_allclose(network.server().x, np.array([2.0]))
    np.testing.assert_allclose(network.server().aux_vars["h"], np.array([0.0]))
    for client in clients:
        np.testing.assert_allclose(client.x, np.array([2.0]))
        np.testing.assert_allclose(client.aux_vars["g"], np.array([0.0]))


def test_feddyn_one_round_update_follows_dynamic_regularization_formula() -> None:
    clients = [Agent(TrackingCost(1.0)), Agent(TrackingCost(3.0))]
    network = FedNetwork(clients=clients)
    algorithm = FedDyn(iterations=1, step_size=1.0, penalty=1.0)
    algorithm.initialize(network)

    network._step(0)  # noqa: SLF001
    algorithm.step(network, 0)

    np.testing.assert_allclose(clients[0].x, np.array([-1.0]))
    np.testing.assert_allclose(clients[1].x, np.array([-3.0]))
    np.testing.assert_allclose(clients[0].aux_vars["g"], np.array([1.0]))
    np.testing.assert_allclose(clients[1].aux_vars["g"], np.array([3.0]))
    np.testing.assert_allclose(network.server().aux_vars["h"], np.array([2.0]))
    np.testing.assert_allclose(network.server().x, np.array([-4.0]))


def test_feddyn_uses_dynamic_state_in_later_local_updates() -> None:
    clients = [Agent(TrackingCost(1.0)), Agent(TrackingCost(3.0))]
    network = FedNetwork(clients=clients)
    algorithm = FedDyn(iterations=2, step_size=1.0, penalty=1.0)
    algorithm.initialize(network)

    network._step(0)  # noqa: SLF001
    algorithm.step(network, 0)
    network._step(1)  # noqa: SLF001
    algorithm.step(network, 1)

    np.testing.assert_allclose(clients[0].x, np.array([-4.0]))
    np.testing.assert_allclose(clients[1].x, np.array([-4.0]))
    np.testing.assert_allclose(clients[0].aux_vars["g"], np.array([1.0]))
    np.testing.assert_allclose(clients[1].aux_vars["g"], np.array([3.0]))
    np.testing.assert_allclose(network.server().aux_vars["h"], np.array([2.0]))
    np.testing.assert_allclose(network.server().x, np.array([-6.0]))


def test_feddyn_partial_participation_leaves_unselected_client_state_unchanged() -> None:
    clients = [Agent(TrackingCost(1.0)), Agent(TrackingCost(3.0))]
    network = FedNetwork(clients=clients)
    algorithm = FedDyn(iterations=1, step_size=1.0, penalty=1.0, selection_scheme=FirstClientSelection())
    algorithm.initialize(network)

    network._step(0)  # noqa: SLF001
    algorithm.step(network, 0)

    np.testing.assert_allclose(clients[0].x, np.array([-1.0]))
    np.testing.assert_allclose(clients[0].aux_vars["g"], np.array([1.0]))
    np.testing.assert_allclose(clients[1].x, np.array([0.0]))
    np.testing.assert_allclose(clients[1].aux_vars["g"], np.array([0.0]))
    np.testing.assert_allclose(network.server().aux_vars["h"], np.array([0.5]))
    np.testing.assert_allclose(network.server().x, np.array([-1.5]))


def test_feddyn_aggregate_uses_only_received_client_models() -> None:
    clients = [Agent(TrackingCost(1.0)), Agent(TrackingCost(2.0))]
    network = FedNetwork(clients=clients)
    algorithm = FedDyn(iterations=1, penalty=1.0)
    algorithm.initialize(network)

    network.send(sender=clients[0], receiver=network.server(), msg=np.array([2.0]))

    algorithm.aggregate(network, clients)

    np.testing.assert_allclose(network.server().aux_vars["h"], np.array([-1.0]))
    np.testing.assert_allclose(network.server().x, np.array([3.0]))


def test_feddyn_aggregate_keeps_server_state_when_no_models_are_received() -> None:
    clients = [Agent(TrackingCost(1.0)), Agent(TrackingCost(2.0))]
    network = FedNetwork(clients=clients)
    algorithm = FedDyn(iterations=1, penalty=1.0)
    algorithm.initialize(network)
    network.server().x = np.array([7.0])
    network.server().aux_vars["h"] = np.array([4.0])

    algorithm.aggregate(network, clients)

    np.testing.assert_allclose(network.server().x, np.array([7.0]))
    np.testing.assert_allclose(network.server().aux_vars["h"], np.array([4.0]))


def test_feddyn_clients_without_server_broadcast_do_not_participate() -> None:
    client = Agent(TrackingCost(1.0))
    server = Agent(ZeroCost((1,)))
    network = FedNetwork(clients=[client], server=server, message_drop={server: DropOnCalls({1}), client: NoDrops()})
    algorithm = FedDyn(iterations=1, step_size=1.0, penalty=1.0)
    algorithm.initialize(network)

    network._step(0)  # noqa: SLF001
    algorithm.step(network, 0)

    np.testing.assert_allclose(network.server().x, np.array([0.0]))
    np.testing.assert_allclose(network.server().aux_vars["h"], np.array([0.0]))
    np.testing.assert_allclose(client.x, np.array([0.0]))
    np.testing.assert_allclose(client.aux_vars["g"], np.array([0.0]))


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    [
        pytest.param({"step_size": 0.0}, "`step_size` must be positive", id="step-size-zero"),
        pytest.param({"penalty": 0.0}, "`penalty` must be positive", id="alpha-zero"),
        pytest.param({"num_local_steps": 0}, "`num_local_steps` must be positive", id="local-epochs-zero"),
    ],
)
def test_feddyn_rejects_invalid_hyperparameters(kwargs: dict[str, float | int], expected_message: str) -> None:
    with pytest.raises(ValueError, match=expected_message):
        FedDyn(iterations=1, **kwargs)
