from typing import Any

import numpy as np
import pytest

from decent_bench.agents import Agent
from decent_bench.costs import Cost
from decent_bench.distributed_algorithms import FedAdagrad, FedAdam, FedAvg, FedProx, FedYogi
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
def test_aggregation_uses_only_received_client_updates(algorithm_cls: type, kwargs: dict[str, float | int]) -> None:
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
def test_default_aggregation_is_uniform(algorithm_cls: type, kwargs: dict[str, float | int]) -> None:
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
    algorithm_cls: type, kwargs: dict[str, float | int]
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
    algorithm_cls: type, kwargs: dict[str, float | int]
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
def test_buffered_stale_client_messages_are_not_aggregated(algorithm_cls: type, kwargs: dict[str, float | int]) -> None:
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


@pytest.mark.parametrize(
    ("algorithm_cls", "kwargs"),
    [
        pytest.param(
            FedAdagrad,
            {"iterations": 1, "step_size": 1.0, "server_step_size": 1.0, "beta_1": 0.0, "tau": 1.0},
            id="fedadagrad",
        ),
        pytest.param(
            FedYogi,
            {
                "iterations": 1,
                "step_size": 1.0,
                "server_step_size": 1.0,
                "beta_1": 0.0,
                "beta_2": 0.25,
                "tau": 1.0,
            },
            id="fedyogi",
        ),
        pytest.param(
            FedAdam,
            {
                "iterations": 1,
                "step_size": 1.0,
                "server_step_size": 1.0,
                "beta_1": 0.0,
                "beta_2": 0.25,
                "tau": 1.0,
            },
            id="fedadam",
        ),
    ],
)
def test_fedopt_aggregation_is_uniform(algorithm_cls: type, kwargs: dict[str, float | int]) -> None:
    algorithm = algorithm_cls(**kwargs)
    network, clients = _make_fed_network(
        TrackingCost(gradient_value=1.0, n_samples=1),
        TrackingCost(gradient_value=3.0, n_samples=3),
    )
    algorithm.initialize(network)

    network.server().x = np.array([1.0])
    network.send(sender=clients[0], receiver=network.server(), msg=np.array([1.0]))
    network.send(sender=clients[1], receiver=network.server(), msg=np.array([3.0]))

    algorithm.aggregate(network, clients)

    np.testing.assert_allclose(network.server().aux_vars["m"], np.array([2.0]))


@pytest.mark.parametrize(
    ("algorithm_cls", "kwargs"),
    [
        pytest.param(
            FedAdagrad,
            {"iterations": 1, "step_size": 1.0, "server_step_size": 1.0, "beta_1": 0.25, "tau": 1.0},
            id="fedadagrad",
        ),
        pytest.param(
            FedYogi,
            {
                "iterations": 1,
                "step_size": 1.0,
                "server_step_size": 1.0,
                "beta_1": 0.25,
                "beta_2": 0.25,
                "tau": 1.0,
            },
            id="fedyogi",
        ),
        pytest.param(
            FedAdam,
            {
                "iterations": 1,
                "step_size": 1.0,
                "server_step_size": 1.0,
                "beta_1": 0.25,
                "beta_2": 0.25,
                "tau": 1.0,
            },
            id="fedadam",
        ),
    ],
)
def test_fedopt_aggregation_keeps_server_state_when_no_updates_are_received(
    algorithm_cls: type, kwargs: dict[str, float | int]
) -> None:
    algorithm = algorithm_cls(**kwargs)
    network, clients = _make_fed_network(TrackingCost(1.0), TrackingCost(2.0))
    algorithm.initialize(network)

    network.server().x = np.array([7.0])
    network.server().aux_vars["m"] = np.array([2.0])
    network.server().aux_vars["v"] = np.array([5.0])

    algorithm.aggregate(network, clients)

    np.testing.assert_allclose(network.server().x, np.array([7.0]))
    np.testing.assert_allclose(network.server().aux_vars["m"], np.array([2.0]))
    np.testing.assert_allclose(network.server().aux_vars["v"], np.array([5.0]))


@pytest.mark.parametrize(
    ("algorithm_cls", "kwargs", "initial_m", "initial_v", "expected_v"),
    [
        pytest.param(
            FedAdagrad,
            {"iterations": 1, "step_size": 1.0, "server_step_size": 2.0, "beta_1": 0.5, "tau": 1.0},
            2.0,
            4.0,
            13.0,
            id="fedadagrad",
        ),
        pytest.param(
            FedYogi,
            {
                "iterations": 1,
                "step_size": 1.0,
                "server_step_size": 2.0,
                "beta_1": 0.5,
                "beta_2": 0.25,
                "tau": 1.0,
            },
            2.0,
            16.0,
            9.25,
            id="fedyogi",
        ),
        pytest.param(
            FedAdam,
            {
                "iterations": 1,
                "step_size": 1.0,
                "server_step_size": 2.0,
                "beta_1": 0.5,
                "beta_2": 0.25,
                "tau": 1.0,
            },
            2.0,
            4.0,
            7.75,
            id="fedadam",
        ),
    ],
)
def test_fedopt_server_state_updates_follow_variant_formula(
    algorithm_cls: type,
    kwargs: dict[str, float | int],
    initial_m: float,
    initial_v: float,
    expected_v: float,
) -> None:
    algorithm = algorithm_cls(**kwargs)
    network, clients = _make_fed_network(TrackingCost(1.0), TrackingCost(2.0))
    algorithm.initialize(network)

    network.server().x = np.array([1.0])
    network.server().aux_vars["m"] = np.array([initial_m])
    network.server().aux_vars["v"] = np.array([initial_v])
    network.send(sender=clients[0], receiver=network.server(), msg=np.array([2.0]))
    network.send(sender=clients[1], receiver=network.server(), msg=np.array([4.0]))

    algorithm.aggregate(network, clients)

    expected_m = np.array([2.5])
    expected_x = np.array([1.0 + (2.0 * expected_m[0] / (np.sqrt(expected_v) + 1.0))])
    np.testing.assert_allclose(network.server().aux_vars["m"], expected_m)
    np.testing.assert_allclose(network.server().aux_vars["v"], np.array([expected_v]))
    np.testing.assert_allclose(network.server().x, expected_x)


@pytest.mark.parametrize(
    ("algorithm_cls", "kwargs"),
    [
        pytest.param(
            FedAdagrad,
            {"iterations": 2, "step_size": 1.0, "server_step_size": 1.0, "beta_1": 0.0, "tau": 1.0},
            id="fedadagrad",
        ),
        pytest.param(
            FedYogi,
            {
                "iterations": 2,
                "step_size": 1.0,
                "server_step_size": 1.0,
                "beta_1": 0.0,
                "beta_2": 0.25,
                "tau": 1.0,
            },
            id="fedyogi",
        ),
        pytest.param(
            FedAdam,
            {
                "iterations": 2,
                "step_size": 1.0,
                "server_step_size": 1.0,
                "beta_1": 0.0,
                "beta_2": 0.25,
                "tau": 1.0,
            },
            id="fedadam",
        ),
    ],
)
def test_fedopt_clients_without_server_broadcast_do_not_participate(
    algorithm_cls: type, kwargs: dict[str, float | int]
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
    first_round_x = np.array(network.server().x, copy=True)
    first_round_m = np.array(network.server().aux_vars["m"], copy=True)
    first_round_v = np.array(network.server().aux_vars["v"], copy=True)

    network._step(1)  # noqa: SLF001
    algorithm.step(network, 1)

    np.testing.assert_allclose(network.server().x, first_round_x)
    np.testing.assert_allclose(network.server().aux_vars["m"], first_round_m)
    np.testing.assert_allclose(network.server().aux_vars["v"], first_round_v)


@pytest.mark.parametrize(
    ("algorithm_cls", "kwargs", "expected_x"),
    [
        pytest.param(
            FedAdagrad,
            {"iterations": 2, "step_size": 1.0, "server_step_size": 1.0, "beta_1": 0.0, "tau": 1.0},
            -0.6666666666666666 + (-3.0 / (np.sqrt(13.0) + 1.0)),
            id="fedadagrad",
        ),
        pytest.param(
            FedYogi,
            {
                "iterations": 2,
                "step_size": 1.0,
                "server_step_size": 1.0,
                "beta_1": 0.0,
                "beta_2": 0.25,
                "tau": 1.0,
            },
            (-2.0 / (np.sqrt(3.0) + 1.0)) + (-3.0 / (np.sqrt(9.75) + 1.0)),
            id="fedyogi",
        ),
        pytest.param(
            FedAdam,
            {
                "iterations": 2,
                "step_size": 1.0,
                "server_step_size": 1.0,
                "beta_1": 0.0,
                "beta_2": 0.25,
                "tau": 1.0,
            },
            (-2.0 / (np.sqrt(3.0) + 1.0)) + (-3.0 / (np.sqrt(7.5) + 1.0)),
            id="fedadam",
        ),
    ],
)
def test_fedopt_buffered_stale_client_messages_are_not_aggregated(
    algorithm_cls: type, kwargs: dict[str, float | int], expected_x: float
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

    network._step(1)  # noqa: SLF001
    algorithm.step(network, 1)

    np.testing.assert_allclose(network.server().x, np.array([expected_x]))


@pytest.mark.parametrize(
    ("algorithm_cls", "kwargs"),
    [
        pytest.param(
            FedAdagrad,
            {"iterations": 1, "step_size": 1.0, "server_step_size": 1.0, "beta_1": 0.0, "tau": 1.0},
            id="fedadagrad",
        ),
        pytest.param(
            FedYogi,
            {
                "iterations": 1,
                "step_size": 1.0,
                "server_step_size": 1.0,
                "beta_1": 0.0,
                "beta_2": 0.25,
                "tau": 1.0,
            },
            id="fedyogi",
        ),
        pytest.param(
            FedAdam,
            {
                "iterations": 1,
                "step_size": 1.0,
                "server_step_size": 1.0,
                "beta_1": 0.0,
                "beta_2": 0.25,
                "tau": 1.0,
            },
            id="fedadam",
        ),
    ],
)
def test_fedopt_step_runs_one_round_end_to_end(algorithm_cls: type, kwargs: dict[str, float | int]) -> None:
    algorithm = algorithm_cls(**kwargs)
    network, clients = _make_fed_network(TrackingCost(1.0), TrackingCost(2.0))
    algorithm.initialize(network)

    network._step(0)  # noqa: SLF001
    algorithm.step(network, 0)

    assert network.server().aux_vars is not None
    assert set(network.server().aux_vars) == {"m", "v"}
    assert clients[0].x.shape == (1,)
    assert clients[1].x.shape == (1,)
    assert not np.allclose(network.server().x, np.zeros(1))


@pytest.mark.parametrize(
    ("algorithm_cls", "kwargs"),
    [
        pytest.param(FedAdagrad, {"iterations": 1, "step_size": 1.0}, id="fedadagrad"),
        pytest.param(FedYogi, {"iterations": 1, "step_size": 1.0}, id="fedyogi"),
        pytest.param(FedAdam, {"iterations": 1, "step_size": 1.0}, id="fedadam"),
    ],
)
def test_fedopt_local_uploads_are_model_deltas(algorithm_cls: type, kwargs: dict[str, float | int]) -> None:
    algorithm = algorithm_cls(**kwargs)
    network, clients = _make_fed_network(TrackingCost(1.0), TrackingCost(2.0))
    algorithm.initialize(network)

    network.server().x = np.array([5.0])
    selected_clients = network.clients()
    algorithm.server_broadcast(network, selected_clients)
    participating_clients = algorithm._clients_with_server_broadcast(network, selected_clients)
    algorithm._clear_buffered_server_messages(network, participating_clients)
    algorithm._run_local_updates(network, participating_clients)

    np.testing.assert_allclose(network.server().messages[clients[0]], np.array([-1.0]))
    np.testing.assert_allclose(network.server().messages[clients[1]], np.array([-2.0]))
