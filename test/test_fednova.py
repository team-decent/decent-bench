from typing import Any

import numpy as np
import pytest

from decent_bench.agents import Agent
from decent_bench.algorithms.federated import FedAvg, FedNova
from decent_bench.costs import EmpiricalRiskCost
from decent_bench.networks import FedNetwork
from decent_bench.schemes import DropScheme, NoDrops
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks


_NORMALIZER_CHANNEL = "normalizer"
_CUMULATIVE_GRADIENT_CHANNEL = "cumulative_gradient"


class TrackingCost(EmpiricalRiskCost):
    def __init__(self, gradient_value: float = 1.0, *, n_samples: int = 1):
        self._gradient = np.array([gradient_value], dtype=float)
        self._n_samples = n_samples
        self._dataset = [(np.array([0.0]), np.array([0.0])) for _ in range(n_samples)]

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

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def batch_size(self) -> int:
        return self._n_samples

    @property
    def dataset(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return self._dataset

    def predict(self, x: np.ndarray, data: list[np.ndarray]) -> np.ndarray:
        del x
        return np.zeros(len(data), dtype=float)

    def function(self, x: np.ndarray, indices: Any = "batch", **kwargs: Any) -> float:  # noqa: ANN401
        del x, kwargs
        self._sample_batch_indices(indices)
        return 0.0

    def _get_batch_data(self, indices: Any = "batch") -> list[tuple[np.ndarray, np.ndarray]]:  # noqa: ANN401
        batch_indices = self._sample_batch_indices(indices)
        return [self._dataset[index] for index in batch_indices]

    def gradient(self, x: np.ndarray, indices: Any = "batch", **kwargs: Any) -> np.ndarray:  # noqa: ANN401
        del x, kwargs
        self._sample_batch_indices(indices)
        return self._gradient.copy()

    def hessian(self, x: np.ndarray, indices: Any = "batch", **kwargs: Any) -> np.ndarray:  # noqa: ANN401
        del x, kwargs
        self._sample_batch_indices(indices)
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


def _make_fed_network(*client_specs: float | tuple[float, int]) -> FedNetwork:
    clients = []
    for i, client_spec in enumerate(client_specs):
        if isinstance(client_spec, tuple):
            gradient_value, n_samples = client_spec
        else:
            gradient_value, n_samples = client_spec, 1
        clients.append(Agent(TrackingCost(gradient_value, n_samples=n_samples)))
    return FedNetwork(clients=clients)


def _run_fed_algorithm(network: FedNetwork, algorithm: FedAvg | FedNova) -> np.ndarray:
    algorithm.initialize(network)
    for iteration in range(algorithm.iterations):
        network._step(iteration)  # noqa: SLF001
        algorithm.step(network, iteration)
    return np.copy(network.server().x)


def _expected_single_client_fednova(
    *,
    gradient_value: float,
    local_steps: int,
    iterations: int,
    step_size: float = 1.0,
    use_momentum: bool = False,
    momentum: float = 0.9,
    use_prox: bool = False,
    penalty: float = 0.0,
    use_server_momentum: bool = False,
    server_momentum: float = 0.9,
) -> float:
    server_x = 0.0
    server_momentum_state = 0.0

    for _ in range(iterations):
        local_x = server_x
        local_momentum = 0.0
        cumulative_gradient = 0.0
        a_i = 0.0
        momentum_scalar = 0.0

        for _ in range(local_steps):
            grad = gradient_value
            if use_prox:
                grad += penalty * (local_x - server_x)

            if use_momentum:
                local_momentum = (momentum * local_momentum) + grad
                direction = local_momentum
            else:
                direction = grad

            local_step_update = step_size * direction
            local_x -= local_step_update
            cumulative_gradient += local_step_update

            momentum_scalar = (momentum * momentum_scalar) + 1.0 if use_momentum else 1.0
            if use_prox:
                a_i = ((1 - (step_size * penalty)) * a_i) + momentum_scalar
            else:
                a_i += momentum_scalar

        tau_eff = a_i
        global_update = (tau_eff / a_i) * cumulative_gradient
        if use_server_momentum:
            server_momentum_state = (server_momentum * server_momentum_state) + global_update
            server_x -= server_momentum_state
        else:
            server_x -= global_update

    return server_x


def test_fednova_supports_heterogeneous_local_steps() -> None:
    network = _make_fed_network(1.0, 3.0)
    clients = network.clients()
    algorithm = FedNova(iterations=1, step_size=1.0, num_local_steps={clients[0]: 1, clients[1]: 3})

    algorithm.initialize(network)
    network._step(0)  # noqa: SLF001
    algorithm.step(network, 0)

    np.testing.assert_allclose(network.server().x, np.array([-4.0]))


def test_plain_fednova_equals_fedavg_when_local_steps_and_aggregation_weights_match() -> None:
    fednova_network = _make_fed_network(1.0, 3.0)
    fedavg_network = _make_fed_network(1.0, 3.0)
    fednova = FedNova(iterations=1, step_size=1.0, num_local_steps=2)
    fedavg = FedAvg(iterations=1, step_size=1.0, num_local_steps=2)

    fednova_server_x = _run_fed_algorithm(fednova_network, fednova)
    fedavg_server_x = _run_fed_algorithm(fedavg_network, fedavg)

    np.testing.assert_allclose(fednova_server_x, np.array([-4.0]))
    np.testing.assert_allclose(fedavg_server_x, np.array([-4.0]))
    np.testing.assert_allclose(fednova_server_x, fedavg_server_x)


def test_fednova_copies_local_step_mapping_on_initialize() -> None:
    network = _make_fed_network(1.0, 3.0)
    clients = network.clients()
    local_steps = {clients[0]: 1, clients[1]: 3}
    algorithm = FedNova(iterations=1, step_size=1.0, num_local_steps=local_steps)

    algorithm.initialize(network)
    local_steps[clients[0]] = 7

    assert isinstance(algorithm.num_local_steps, dict)
    assert algorithm.num_local_steps[clients[0]] == 1
    assert algorithm.num_local_steps[clients[1]] == 3


def test_fednova_normalizes_scalar_local_steps_to_client_mapping_on_initialize() -> None:
    network = _make_fed_network(1.0, 3.0)
    clients = network.clients()
    algorithm = FedNova(iterations=1, step_size=1.0, num_local_steps=2)

    algorithm.initialize(network)

    assert algorithm.num_local_steps == {clients[0]: 2, clients[1]: 2}


def test_fednova_resolves_client_sample_counts_once_on_initialize(monkeypatch: pytest.MonkeyPatch) -> None:
    network = _make_fed_network((1.0, 1), (3.0, 3))
    algorithm = FedNova(iterations=1, step_size=1.0, num_local_steps=1)
    infer_client_data_size_calls = 0

    def _tracking_infer_client_data_size(client: Agent) -> float:
        nonlocal infer_client_data_size_calls
        infer_client_data_size_calls += 1
        return float(client.cost.n_samples)  # type: ignore[attr-defined]

    monkeypatch.setattr(
        "decent_bench.algorithms.federated._fed_nova.infer_client_data_size",
        _tracking_infer_client_data_size,
    )

    algorithm.initialize(network)
    assert infer_client_data_size_calls == len(network.clients())

    network._step(0)  # noqa: SLF001
    algorithm.step(network, 0)

    assert infer_client_data_size_calls == len(network.clients())


@pytest.mark.parametrize(
    ("test_id", "iterations", "local_steps", "kwargs"),
    [
        pytest.param(
            "local-momentum",
            1,
            2,
            {"use_momentum": True, "momentum": 0.5},
            id="only-local-momentum",
        ),
        pytest.param(
            "prox",
            1,
            2,
            {"use_prox": True, "penalty": 0.5},
            id="only-prox",
        ),
        pytest.param(
            "server-momentum",
            2,
            1,
            {"use_server_momentum": True, "server_momentum": 0.5},
            id="only-server-momentum",
        ),
        pytest.param(
            "both-momentums",
            2,
            2,
            {"use_momentum": True, "momentum": 0.5, "use_server_momentum": True, "server_momentum": 0.5},
            id="both-momentums",
        ),
        pytest.param(
            "server-momentum-prox",
            2,
            2,
            {"use_prox": True, "penalty": 0.5, "use_server_momentum": True, "server_momentum": 0.5},
            id="server-momentum-plus-prox",
        ),
        pytest.param(
            "momentum-prox",
            1,
            2,
            {"use_momentum": True, "momentum": 0.5, "use_prox": True, "penalty": 0.5},
            id="local-momentum-plus-prox",
        ),
        pytest.param(
            "all-three",
            2,
            2,
            {
                "use_momentum": True,
                "momentum": 0.5,
                "use_prox": True,
                "penalty": 0.5,
                "use_server_momentum": True,
                "server_momentum": 0.5,
            },
            id="all-three",
        ),
    ],
)
def test_fednova_matches_scalar_pseudocode_for_option_combinations(
    test_id: str,
    iterations: int,
    local_steps: int,
    kwargs: dict[str, float | bool],
) -> None:
    del test_id
    network = _make_fed_network(1.0)
    algorithm = FedNova(iterations=iterations, step_size=1.0, num_local_steps=local_steps, **kwargs)

    expected_server_x = _expected_single_client_fednova(
        gradient_value=1.0,
        iterations=iterations,
        local_steps=local_steps,
        step_size=1.0,
        use_momentum=bool(kwargs.get("use_momentum", False)),
        momentum=float(kwargs.get("momentum", 0.9)),
        use_prox=bool(kwargs.get("use_prox", False)),
        penalty=float(kwargs.get("penalty", 0.0)),
        use_server_momentum=bool(kwargs.get("use_server_momentum", False)),
        server_momentum=float(kwargs.get("server_momentum", 0.9)),
    )

    actual_server_x = _run_fed_algorithm(network, algorithm)

    np.testing.assert_allclose(actual_server_x, np.array([expected_server_x]))


def test_fednova_uses_data_proportional_client_weights() -> None:
    network = _make_fed_network((1.0, 1), (3.0, 3))
    algorithm = FedNova(iterations=1, step_size=1.0, num_local_steps=1)

    algorithm.initialize(network)
    network._step(0)  # noqa: SLF001
    algorithm.step(network, 0)

    np.testing.assert_allclose(network.server().x, np.array([-2.5]))


def test_fednova_uploads_normalizer_then_cumulative_gradient() -> None:
    network = _make_fed_network((2.0, 1), (4.0, 3))
    clients = network.clients()
    algorithm = FedNova(iterations=1, step_size=1.0, num_local_steps={clients[0]: 2, clients[1]: 1})

    algorithm.initialize(network)
    network._step(0)  # noqa: SLF001
    selected_clients = network.clients()
    algorithm.server_broadcast(network, selected_clients)
    participating_clients = algorithm._clients_with_server_broadcast(network, selected_clients)
    algorithm._run_local_updates(network, participating_clients)

    np.testing.assert_allclose(network.server().message(clients[0], _NORMALIZER_CHANNEL), np.array([2.0]))
    np.testing.assert_allclose(network.server().message(clients[1], _NORMALIZER_CHANNEL), np.array([1.0]))
    np.testing.assert_allclose(network.server().message(clients[0], _CUMULATIVE_GRADIENT_CHANNEL), np.array([4.0]))
    np.testing.assert_allclose(network.server().message(clients[1], _CUMULATIVE_GRADIENT_CHANNEL), np.array([4.0]))


def test_fednova_stores_client_sample_counts_on_server_initialize() -> None:
    network = _make_fed_network((2.0, 1), (4.0, 3))
    clients = network.clients()
    algorithm = FedNova(iterations=1, step_size=1.0, num_local_steps=1)

    algorithm.initialize(network)

    assert network.server().aux_vars["client_sample_counts"] == {clients[0]: 1.0, clients[1]: 3.0}


def test_fednova_renormalizes_client_weights_over_received_subset() -> None:
    network = _make_fed_network((1.0, 1), (10.0, 3))
    clients = network.clients()
    algorithm = FedNova(iterations=1, step_size=1.0, num_local_steps=1)

    algorithm.initialize(network)
    network._step(0)  # noqa: SLF001
    selected_clients = network.clients()
    algorithm.server_broadcast(network, selected_clients)
    participating_clients = algorithm._clients_with_server_broadcast(network, selected_clients)
    algorithm._run_local_updates(network, participating_clients)
    network.server()._received_messages.clear(  # noqa: SLF001
        sender=clients[1],
        channel=_CUMULATIVE_GRADIENT_CHANNEL,
    )

    algorithm.aggregate(network, participating_clients)

    np.testing.assert_allclose(network.server().x, np.array([-1.0]))


def test_fednova_aggregate_rejects_non_positive_normalizer() -> None:
    network = _make_fed_network((2.0, 1), (4.0, 3))
    clients = network.clients()
    algorithm = FedNova(iterations=1, step_size=1.0, num_local_steps=1)

    algorithm.initialize(network)
    network._step(0)  # noqa: SLF001
    selected_clients = network.clients()
    algorithm.server_broadcast(network, selected_clients)
    participating_clients = algorithm._clients_with_server_broadcast(network, selected_clients)
    algorithm._run_local_updates(network, participating_clients)
    network.send(
        sender=clients[0],
        receiver=network.server(),
        msg=np.array([0.0]),
        channel=_NORMALIZER_CHANNEL,
    )

    with pytest.raises(ValueError, match="FedNova coefficients `a_i` must be positive"):
        algorithm.aggregate(network, participating_clients)


def test_fednova_skips_round_when_all_normalizer_uploads_are_dropped() -> None:
    clients = [Agent(TrackingCost(1.0, n_samples=1)), Agent(TrackingCost(10.0, n_samples=3))]
    server = Agent(TrackingCost(0.0))
    network = FedNetwork(
        clients=clients,
        server=server,
        message_drop={server: NoDrops(), clients[0]: DropOnCalls({1}), clients[1]: DropOnCalls({1})},
    )
    algorithm = FedNova(iterations=1, step_size=1.0, num_local_steps=1)

    algorithm.initialize(network)
    network._step(0)  # noqa: SLF001
    algorithm.step(network, 0)

    np.testing.assert_allclose(network.server().x, np.array([0.0]))
    assert network.server().messages(_NORMALIZER_CHANNEL) == {}
    assert set(network.server().messages(_CUMULATIVE_GRADIENT_CHANNEL)) == set(clients)


@pytest.mark.parametrize("dropped_calls", [{1}, {2}])
def test_fednova_uses_only_clients_with_both_uploads(dropped_calls: set[int]) -> None:
    clients = [Agent(TrackingCost(1.0, n_samples=1)), Agent(TrackingCost(10.0, n_samples=3))]
    server = Agent(TrackingCost(0.0))
    network = FedNetwork(
        clients=clients,
        server=server,
        message_drop={server: NoDrops(), clients[0]: DropOnCalls(dropped_calls), clients[1]: NoDrops()},
    )
    algorithm = FedNova(iterations=1, step_size=1.0, num_local_steps=1)

    algorithm.initialize(network)
    network._step(0)  # noqa: SLF001
    algorithm.step(network, 0)

    np.testing.assert_allclose(network.server().x, np.array([-10.0]))


def test_fednova_differs_from_fedavg_when_local_steps_are_heterogeneous() -> None:
    fednova_network = _make_fed_network(1.0, 3.0)
    fedavg_network = _make_fed_network(1.0, 3.0)
    fednova_clients = fednova_network.clients()
    fednova = FedNova(
        iterations=1,
        step_size=1.0,
        num_local_steps={fednova_clients[0]: 1, fednova_clients[1]: 3},
    )
    fedavg = FedAvg(iterations=1, step_size=1.0, num_local_steps=1)

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
        num_local_steps={fednova_clients[0]: 1, fednova_clients[1]: 3},
    )
    fedavg = FedAvg(iterations=1, step_size=1.0, num_local_steps=1)

    fednova.initialize(fednova_network)
    fedavg.initialize(fedavg_network)

    fednova_network._step(0)  # noqa: SLF001
    fedavg_network._step(0)  # noqa: SLF001

    fednova.step(fednova_network, 0)
    fedavg.step(fedavg_network, 0)

    np.testing.assert_allclose(fednova_network.server().x, np.array([-6.25]))
    np.testing.assert_allclose(fedavg_network.server().x, np.array([-2.0]))
    assert not np.allclose(fednova_network.server().x, fedavg_network.server().x)


def test_fednova_rejects_sequence_num_local_steps() -> None:
    with pytest.raises(TypeError, match="`num_local_steps` must be an int or a mapping from Agent"):
        FedNova(iterations=1, step_size=1.0, num_local_steps=[1, 3])


@pytest.mark.parametrize("num_local_steps", [0, -1])
def test_fednova_rejects_invalid_scalar_num_local_steps(num_local_steps: int) -> None:
    with pytest.raises(ValueError, match="`num_local_steps` must be positive"):
        FedNova(iterations=1, step_size=1.0, num_local_steps=num_local_steps)


@pytest.mark.parametrize("num_local_steps", [1.5])
def test_fednova_rejects_non_integer_scalar_num_local_steps(num_local_steps: object) -> None:
    with pytest.raises(TypeError, match="`num_local_steps` must be an int or a mapping from Agent"):
        FedNova(iterations=1, step_size=1.0, num_local_steps=num_local_steps)


@pytest.mark.parametrize("num_local_steps", [{}, {"not-an-agent": 1}, {1: 1}])
def test_fednova_rejects_local_step_mappings_missing_network_clients(num_local_steps: object) -> None:
    network = _make_fed_network(1.0, 3.0)
    algorithm = FedNova(iterations=1, step_size=1.0, num_local_steps=num_local_steps)

    with pytest.raises(ValueError, match="`num_local_steps` mapping must provide a value for every network client"):
        algorithm.initialize(network)


@pytest.mark.parametrize("step_value", [0, -1])
def test_fednova_rejects_invalid_local_step_mapping_values(step_value: float) -> None:
    client = Agent(TrackingCost())
    with pytest.raises(ValueError, match="`num_local_steps` must have positive values"):
        FedNova(iterations=1, step_size=1.0, num_local_steps={client: step_value})


@pytest.mark.parametrize("step_value", [2.0])
def test_fednova_rejects_non_integer_local_step_mapping_values(step_value: object) -> None:
    client = Agent(TrackingCost())
    with pytest.raises(TypeError, match="`num_local_steps` mapping values must be integers"):
        FedNova(iterations=1, step_size=1.0, num_local_steps={client: step_value})


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    [
        pytest.param({"momentum": -0.1}, "`momentum` must satisfy 0 <= momentum < 1", id="momentum-negative"),
        pytest.param({"momentum": 1.0}, "`momentum` must satisfy 0 <= momentum < 1", id="momentum-too-large"),
        pytest.param({"penalty": -0.1}, "`penalty` must be non-negative", id="penalty-negative"),
        pytest.param({"server_momentum": -0.1}, "`server_momentum` must satisfy 0 <= server_momentum < 1", id="server_momentum-negative"),
        pytest.param({"server_momentum": 1.0}, "`server_momentum` must satisfy 0 <= server_momentum < 1", id="server_momentum-too-large"),
    ],
)
def test_fednova_rejects_invalid_hyperparameters(kwargs: dict[str, float], expected_message: str) -> None:
    with pytest.raises(ValueError, match=expected_message):
        FedNova(iterations=1, step_size=1.0, num_local_steps=1, **kwargs)


def test_fednova_rejects_local_step_mappings_that_do_not_match_network_clients() -> None:
    network = _make_fed_network(1.0, 3.0)
    other_network = _make_fed_network(2.0)
    algorithm = FedNova(
        iterations=1,
        step_size=1.0,
        num_local_steps={network.clients()[0]: 1, other_network.clients()[0]: 3},
    )

    with pytest.raises(ValueError, match="`num_local_steps` mapping must provide a value for every network client"):
        algorithm.initialize(network)


def test_fednova_rejects_local_step_mappings_missing_clients() -> None:
    network = _make_fed_network(1.0, 3.0)
    clients = network.clients()
    algorithm = FedNova(iterations=1, step_size=1.0, num_local_steps={clients[0]: 1})

    with pytest.raises(ValueError, match="missing clients"):
        algorithm.initialize(network)


def test_fednova_ignores_local_step_mappings_for_unknown_clients() -> None:
    network = _make_fed_network(1.0, 3.0)
    other_network = _make_fed_network(2.0)
    clients = network.clients()
    algorithm = FedNova(
        iterations=1,
        step_size=1.0,
        num_local_steps={clients[0]: 1, clients[1]: 3, other_network.clients()[0]: 2},
    )

    algorithm.initialize(network)

    assert algorithm.num_local_steps == {clients[0]: 1, clients[1]: 3}
