from typing import Any

import numpy as np
import pytest

import decent_bench.utils.interoperability as iop
from decent_bench.agents import Agent
from decent_bench.algorithms.federated import FedPD
from decent_bench.costs import Cost, ZeroCost
from decent_bench.networks import FedNetwork
from decent_bench.schemes import DropScheme, NoDrops
from decent_bench.utils.types import (
    SupportedDevices,
    SupportedFrameworks,
)


_CENTER_CANDIDATE_LABEL = "center_candidate"
_CENTER_UPDATE_LABEL = "center_update"


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


def test_fedpd_initializes_primal_dual_and_center_states() -> None:
    clients = [Agent(TrackingCost(1.0)), Agent(TrackingCost(2.0))]
    network = FedNetwork(clients=clients)
    algorithm = FedPD(iterations=1, x0=np.array([2.0]))

    algorithm.initialize(network)

    np.testing.assert_allclose(network.server().x, np.array([2.0]))
    for client in clients:
        np.testing.assert_allclose(client.x, np.array([2.0]))
        np.testing.assert_allclose(client.aux_vars["lambda"], np.array([0.0]))
        np.testing.assert_allclose(client.aux_vars["center"], np.array([2.0]))


def test_fedpd_p_zero_always_aggregates_and_synchronizes_centers() -> None:
    clients = [Agent(TrackingCost(1.0)), Agent(TrackingCost(3.0))]
    network = FedNetwork(clients=clients)
    algorithm = FedPD(iterations=1, step_size=1.0, eta=1.0, skip_probability=0.0)
    algorithm.initialize(network)

    network._step(0)  # noqa: SLF001
    algorithm.step(network, 0)

    np.testing.assert_allclose(network.server().x, np.array([-4.0]))
    np.testing.assert_allclose(clients[0].x, np.array([-1.0]))
    np.testing.assert_allclose(clients[1].x, np.array([-3.0]))
    np.testing.assert_allclose(clients[0].aux_vars["lambda"], np.array([-1.0]))
    np.testing.assert_allclose(clients[1].aux_vars["lambda"], np.array([-3.0]))
    np.testing.assert_allclose(clients[0].aux_vars["center"], np.array([-4.0]))
    np.testing.assert_allclose(clients[1].aux_vars["center"], np.array([-4.0]))


def test_fedpd_supports_heterogeneous_local_steps() -> None:
    clients = [Agent(TrackingCost(1.0)), Agent(TrackingCost(3.0))]
    network = FedNetwork(clients=clients)
    algorithm = FedPD(
        iterations=1,
        step_size=0.5,
        eta=1.0,
        skip_probability=0.0,
        num_local_steps={clients[0]: 1, clients[1]: 2},
    )
    algorithm.initialize(network)

    network._step(0)  # noqa: SLF001
    algorithm.step(network, 0)

    np.testing.assert_allclose(clients[0].x, np.array([-0.5]))
    np.testing.assert_allclose(clients[1].x, np.array([-2.25]))
    np.testing.assert_allclose(network.server().x, np.array([-2.75]))


def test_fedpd_always_selects_all_active_clients() -> None:
    clients = [Agent(TrackingCost(1.0)), Agent(TrackingCost(2.0)), Agent(TrackingCost(3.0))]
    network = FedNetwork(clients=clients)
    algorithm = FedPD(iterations=1, step_size=1.0, eta=1.0, skip_probability=1.0)
    algorithm.initialize(network)

    network._step(0)  # noqa: SLF001
    algorithm.step(network, 0)

    np.testing.assert_allclose(clients[0].x, np.array([-1.0]))
    np.testing.assert_allclose(clients[1].x, np.array([-2.0]))
    np.testing.assert_allclose(clients[2].x, np.array([-3.0]))


def test_fedpd_rejects_selection_scheme_argument() -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument 'selection_scheme'"):
        FedPD(selection_scheme=object())


def test_fedpd_p_one_always_skips_aggregation() -> None:
    clients = [Agent(TrackingCost(1.0)), Agent(TrackingCost(3.0))]
    network = FedNetwork(clients=clients)
    algorithm = FedPD(iterations=1, step_size=1.0, eta=1.0, skip_probability=1.0)
    algorithm.initialize(network)

    network._step(0)  # noqa: SLF001
    algorithm.step(network, 0)

    np.testing.assert_allclose(network.server().x, np.array([0.0]))
    np.testing.assert_allclose(clients[0].aux_vars["center"], np.array([-2.0]))
    np.testing.assert_allclose(clients[1].aux_vars["center"], np.array([-6.0]))
    assert network.server().messages() == {}


def test_fedpd_dual_update_uses_previous_center() -> None:
    algorithm = FedPD(iterations=1, step_size=1.0, eta=2.0, skip_probability=1.0)
    client = Agent(TrackingCost(gradient_value=0.0))
    network = FedNetwork(clients=[client])
    algorithm.initialize(network)
    client.x = np.array([1.0])
    client.aux_vars["lambda"] = np.array([0.5])
    client.aux_vars["center"] = np.array([2.0])

    algorithm._run_local_updates([client])

    np.testing.assert_allclose(client.x, np.array([1.0]))
    np.testing.assert_allclose(client.aux_vars["lambda"], np.array([0.0]))
    np.testing.assert_allclose(client.aux_vars["center"], np.array([1.0]))


def test_fedpd_aggregate_uses_only_received_center_candidates() -> None:
    clients = [Agent(TrackingCost(1.0)), Agent(TrackingCost(2.0))]
    network = FedNetwork(clients=clients)
    algorithm = FedPD(iterations=1)
    algorithm.initialize(network)
    network.server().x = np.array([10.0])

    network.send(
        sender=clients[0],
        receiver=network.server(),
        msg=np.array([2.0]),
        label=_CENTER_CANDIDATE_LABEL,
    )

    algorithm.aggregate(network, clients)

    np.testing.assert_allclose(network.server().x, np.array([2.0]))


def test_fedpd_synchronizes_all_active_clients_after_aggregating_received_candidates() -> None:
    clients = [Agent(TrackingCost(1.0)), Agent(TrackingCost(3.0))]
    network = FedNetwork(
        clients=clients,
        message_drop={clients[0]: DropOnCalls({1}), clients[1]: NoDrops()},
    )
    algorithm = FedPD(iterations=1, step_size=1.0, eta=1.0, skip_probability=0.0)
    algorithm.initialize(network)

    network._step(0)  # noqa: SLF001
    algorithm.step(network, 0)

    np.testing.assert_allclose(network.server().x, np.array([-6.0]))
    np.testing.assert_allclose(clients[0].aux_vars["center"], np.array([-6.0]))
    np.testing.assert_allclose(clients[1].aux_vars["center"], np.array([-6.0]))


def test_fedpd_does_not_synchronize_when_no_center_candidates_are_received() -> None:
    clients = [Agent(TrackingCost(1.0)), Agent(TrackingCost(3.0))]
    network = FedNetwork(
        clients=clients,
        message_drop={clients[0]: DropOnCalls({1}), clients[1]: DropOnCalls({1})},
    )
    algorithm = FedPD(iterations=1, step_size=1.0, eta=1.0, skip_probability=0.0)
    algorithm.initialize(network)

    network._step(0)  # noqa: SLF001
    algorithm.step(network, 0)

    np.testing.assert_allclose(network.server().x, np.array([0.0]))
    np.testing.assert_allclose(clients[0].aux_vars["center"], np.array([-2.0]))
    np.testing.assert_allclose(clients[1].aux_vars["center"], np.array([-6.0]))
    assert network.server() not in clients[0].messages(_CENTER_UPDATE_LABEL)
    assert network.server() not in clients[1].messages(_CENTER_UPDATE_LABEL)


def test_fedpd_keeps_local_center_when_server_sync_message_is_dropped() -> None:
    clients = [Agent(TrackingCost(1.0)), Agent(TrackingCost(3.0))]
    server = Agent(ZeroCost((1,)))
    network = FedNetwork(
        clients=clients,
        server=server,
        message_drop={server: DropOnCalls({1}), clients[0]: NoDrops(), clients[1]: NoDrops()},
    )
    algorithm = FedPD(iterations=1, step_size=1.0, eta=1.0, skip_probability=0.0)
    algorithm.initialize(network)

    network._step(0)  # noqa: SLF001
    algorithm.step(network, 0)

    np.testing.assert_allclose(network.server().x, np.array([-4.0]))
    np.testing.assert_allclose(clients[0].aux_vars["center"], np.array([-2.0]))
    np.testing.assert_allclose(clients[1].aux_vars["center"], np.array([-4.0]))


def test_fedpd_probabilistic_communication_is_reproducible_when_seeded() -> None:
    def run_once() -> tuple[np.ndarray, list[np.ndarray]]:
        iop.set_seed(123, frameworks=[])
        clients = [Agent(TrackingCost(1.0)), Agent(TrackingCost(3.0))]
        network = FedNetwork(clients=clients)
        algorithm = FedPD(iterations=5, step_size=1.0, eta=1.0, skip_probability=0.5)
        algorithm.initialize(network)
        for iteration in range(algorithm.iterations):
            network._step(iteration)  # noqa: SLF001
            algorithm.step(network, iteration)
        return np.copy(network.server().x), [np.copy(client.aux_vars["center"]) for client in clients]

    server_x_1, centers_1 = run_once()
    server_x_2, centers_2 = run_once()

    np.testing.assert_allclose(server_x_1, server_x_2)
    for center_1, center_2 in zip(centers_1, centers_2, strict=True):
        np.testing.assert_allclose(center_1, center_2)


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    [
        pytest.param({"step_size": 0.0}, "`step_size` must be positive", id="step-size-zero"),
        pytest.param({"eta": 0.0}, "`eta` must be positive", id="eta-zero"),
        pytest.param(
            {"skip_probability": -0.1},
            "`skip_probability` must satisfy 0 <= skip_probability <= 1",
            id="skip-probability-negative",
        ),
        pytest.param(
            {"skip_probability": 1.1},
            "`skip_probability` must satisfy 0 <= skip_probability <= 1",
            id="skip-probability-too-large",
        ),
        pytest.param({"num_local_steps": 0}, "`num_local_steps` must be positive", id="local-steps-zero"),
    ],
)
def test_fedpd_rejects_invalid_hyperparameters(kwargs: dict[str, float | int], expected_message: str) -> None:
    with pytest.raises(ValueError, match=expected_message):
        FedPD(iterations=1, **kwargs)


@pytest.mark.parametrize("num_local_steps", [1.5])
def test_fedpd_rejects_non_integer_scalar_num_local_steps(num_local_steps: object) -> None:
    with pytest.raises(TypeError, match="`num_local_steps` must be an int or a mapping from Agent"):
        FedPD(iterations=1, num_local_steps=num_local_steps)


def test_fedpd_normalizes_scalar_local_steps_to_client_mapping_on_initialize() -> None:
    clients = [Agent(TrackingCost(1.0)), Agent(TrackingCost(2.0))]
    network = FedNetwork(clients=clients)
    algorithm = FedPD(iterations=1, num_local_steps=2)

    algorithm.initialize(network)

    assert algorithm.num_local_steps == {clients[0]: 2, clients[1]: 2}


@pytest.mark.parametrize("num_local_steps", [{}, {"not-an-agent": 1}, {1: 1}])
def test_fedpd_rejects_local_step_mappings_missing_network_clients(num_local_steps: object) -> None:
    network = FedNetwork(clients=[Agent(TrackingCost(1.0)), Agent(TrackingCost(2.0))])
    algorithm = FedPD(iterations=1, num_local_steps=num_local_steps)

    with pytest.raises(ValueError, match="`num_local_steps` mapping must provide a value for every network client"):
        algorithm.initialize(network)


@pytest.mark.parametrize("step_value", [0, -1])
def test_fedpd_rejects_invalid_local_step_mapping_values(step_value: float) -> None:
    client = Agent(TrackingCost())
    with pytest.raises(ValueError, match="`num_local_steps` must have positive values"):
        FedPD(iterations=1, num_local_steps={client: step_value})


@pytest.mark.parametrize("step_value", [2.0])
def test_fedpd_rejects_non_integer_local_step_mapping_values(step_value: object) -> None:
    client = Agent(TrackingCost())
    with pytest.raises(TypeError, match="`num_local_steps` mapping values must be integers"):
        FedPD(iterations=1, num_local_steps={client: step_value})
