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
    beta: float = 0.9,
    use_prox: bool = False,
    mu: float = 0.0,
    use_server_momentum: bool = False,
    gamma: float = 0.9,
) -> float:
    server_x = 0.0
    server_momentum = 0.0

    for _ in range(iterations):
        local_x = server_x
        local_momentum = 0.0
        cumulative_gradient = 0.0
        a_i = 0.0
        momentum_scalar = 0.0

        for _ in range(local_steps):
            grad = gradient_value
            if use_prox:
                grad += mu * (local_x - server_x)

            if use_momentum:
                local_momentum = (beta * local_momentum) + grad
                direction = local_momentum
            else:
                direction = grad

            local_step_update = step_size * direction
            local_x -= local_step_update
            cumulative_gradient += local_step_update

            momentum_scalar = (beta * momentum_scalar) + 1.0 if use_momentum else 1.0
            if use_prox:
                a_i = ((1 - (step_size * mu)) * a_i) + momentum_scalar
            else:
                a_i += momentum_scalar

        tau_eff = a_i
        global_update = (tau_eff / a_i) * cumulative_gradient
        if use_server_momentum:
            server_momentum = (gamma * server_momentum) + global_update
            server_x -= server_momentum
        else:
            server_x -= global_update

    return server_x


def test_fednova_supports_heterogeneous_local_steps() -> None:
    network = _make_fed_network(1.0, 3.0)
    clients = network.clients()
    algorithm = FedNova(iterations=1, step_size=1.0, local_steps={clients[0]: 1, clients[1]: 3})

    algorithm.initialize(network)
    network._step(0)  # noqa: SLF001
    algorithm.step(network, 0)

    np.testing.assert_allclose(network.server().x, np.array([-4.0]))


def test_plain_fednova_equals_fedavg_when_all_clients_use_the_same_local_steps() -> None:
    fednova_network = _make_fed_network(1.0, 3.0)
    fedavg_network = _make_fed_network(1.0, 3.0)
    fednova = FedNova(iterations=1, step_size=1.0, local_steps=2)
    fedavg = FedAvg(iterations=1, step_size=1.0, num_local_epochs=2)

    fednova_server_x = _run_fed_algorithm(fednova_network, fednova)
    fedavg_server_x = _run_fed_algorithm(fedavg_network, fedavg)

    np.testing.assert_allclose(fednova_server_x, np.array([-4.0]))
    np.testing.assert_allclose(fedavg_server_x, np.array([-4.0]))
    np.testing.assert_allclose(fednova_server_x, fedavg_server_x)


def test_fednova_copies_local_step_mapping_on_initialize() -> None:
    network = _make_fed_network(1.0, 3.0)
    clients = network.clients()
    local_steps = {clients[0]: 1, clients[1]: 3}
    algorithm = FedNova(iterations=1, step_size=1.0, local_steps=local_steps)

    algorithm.initialize(network)
    local_steps[clients[0]] = 7

    assert algorithm._local_steps_by_client[clients[0]] == 1  # noqa: SLF001
    assert algorithm._local_steps_by_client[clients[1]] == 3  # noqa: SLF001


@pytest.mark.parametrize(
    ("test_id", "iterations", "local_steps", "kwargs"),
    [
        pytest.param(
            "local-momentum",
            1,
            2,
            {"use_momentum": True, "beta": 0.5},
            id="only-local-momentum",
        ),
        pytest.param(
            "prox",
            1,
            2,
            {"use_prox": True, "mu": 0.5},
            id="only-prox",
        ),
        pytest.param(
            "server-momentum",
            2,
            1,
            {"use_server_momentum": True, "gamma": 0.5},
            id="only-server-momentum",
        ),
        pytest.param(
            "both-momentums",
            2,
            2,
            {"use_momentum": True, "beta": 0.5, "use_server_momentum": True, "gamma": 0.5},
            id="both-momentums",
        ),
        pytest.param(
            "server-momentum-prox",
            2,
            2,
            {"use_prox": True, "mu": 0.5, "use_server_momentum": True, "gamma": 0.5},
            id="server-momentum-plus-prox",
        ),
        pytest.param(
            "momentum-prox",
            1,
            2,
            {"use_momentum": True, "beta": 0.5, "use_prox": True, "mu": 0.5},
            id="local-momentum-plus-prox",
        ),
        pytest.param(
            "all-three",
            2,
            2,
            {
                "use_momentum": True,
                "beta": 0.5,
                "use_prox": True,
                "mu": 0.5,
                "use_server_momentum": True,
                "gamma": 0.5,
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
    algorithm = FedNova(iterations=iterations, step_size=1.0, local_steps=local_steps, **kwargs)

    expected_server_x = _expected_single_client_fednova(
        gradient_value=1.0,
        iterations=iterations,
        local_steps=local_steps,
        step_size=1.0,
        use_momentum=bool(kwargs.get("use_momentum", False)),
        beta=float(kwargs.get("beta", 0.9)),
        use_prox=bool(kwargs.get("use_prox", False)),
        mu=float(kwargs.get("mu", 0.0)),
        use_server_momentum=bool(kwargs.get("use_server_momentum", False)),
        gamma=float(kwargs.get("gamma", 0.9)),
    )

    actual_server_x = _run_fed_algorithm(network, algorithm)

    np.testing.assert_allclose(actual_server_x, np.array([expected_server_x]))


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


@pytest.mark.parametrize("local_steps", [0, -1])
def test_fednova_rejects_invalid_scalar_local_steps(local_steps: int) -> None:
    with pytest.raises(ValueError, match="`local_steps` must be positive"):
        FedNova(iterations=1, step_size=1.0, local_steps=local_steps)


@pytest.mark.parametrize("local_steps", [{}, {"not-an-agent": 1}, {1: 1}])
def test_fednova_rejects_invalid_local_step_mapping_keys(local_steps: object) -> None:
    expected_message = (
        "`local_steps` mapping must be non-empty"
        if local_steps == {}
        else "`local_steps` mapping keys must be Agent instances"
    )
    with pytest.raises((TypeError, ValueError), match=expected_message):
        FedNova(iterations=1, step_size=1.0, local_steps=local_steps)


@pytest.mark.parametrize("step_value", [0, -1, 1.5])
def test_fednova_rejects_invalid_local_step_mapping_values(step_value: float) -> None:
    client = Agent(0, TrackingCost())
    with pytest.raises(ValueError, match="`local_steps` must"):
        FedNova(iterations=1, step_size=1.0, local_steps={client: step_value})


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    [
        pytest.param({"beta": -0.1}, "`beta` must satisfy 0 <= beta < 1", id="beta-negative"),
        pytest.param({"beta": 1.0}, "`beta` must satisfy 0 <= beta < 1", id="beta-too-large"),
        pytest.param({"mu": -0.1}, "`mu` must be non-negative", id="mu-negative"),
        pytest.param({"gamma": -0.1}, "`gamma` must satisfy 0 <= gamma < 1", id="gamma-negative"),
        pytest.param({"gamma": 1.0}, "`gamma` must satisfy 0 <= gamma < 1", id="gamma-too-large"),
    ],
)
def test_fednova_rejects_invalid_hyperparameters(kwargs: dict[str, float], expected_message: str) -> None:
    with pytest.raises(ValueError, match=expected_message):
        FedNova(iterations=1, step_size=1.0, local_steps=1, **kwargs)


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


def test_fednova_rejects_local_step_mappings_missing_clients() -> None:
    network = _make_fed_network(1.0, 3.0)
    clients = network.clients()
    algorithm = FedNova(iterations=1, step_size=1.0, local_steps={clients[0]: 1})

    with pytest.raises(ValueError, match="missing clients"):
        algorithm.initialize(network)


def test_fednova_rejects_local_step_mappings_with_unknown_clients() -> None:
    network = _make_fed_network(1.0, 3.0)
    other_network = _make_fed_network(2.0)
    clients = network.clients()
    algorithm = FedNova(
        iterations=1,
        step_size=1.0,
        local_steps={clients[0]: 1, clients[1]: 3, other_network.clients()[0]: 2},
    )

    with pytest.raises(ValueError, match="unexpected clients"):
        algorithm.initialize(network)
