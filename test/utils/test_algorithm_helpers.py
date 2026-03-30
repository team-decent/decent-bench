import networkx as nx
import numpy as np
import pytest

import decent_bench.utils.interoperability as iop
from decent_bench.agents import Agent
from decent_bench.costs import L2RegularizerCost, PyTorchCost
from decent_bench.networks import FedNetwork, P2PNetwork
from decent_bench.utils.algorithm_helpers import (
    infer_client_weight,
    initial_states,
    normal_initialization,
    pytorch_initialization,
    uniform_initialization,
)
from decent_bench.utils.array import Array
from decent_bench.utils.pytorch_utils import SimpleLinearModel


def _make_p2p_network(n_agents: int = 3, shape: tuple[int, ...] = (2,)) -> P2PNetwork:
    agents = [Agent(i, L2RegularizerCost(shape)) for i in range(n_agents)]
    return P2PNetwork(graph=nx.complete_graph(n_agents), agents=agents)


def test_initial_states_none_builds_zero_state_per_agent() -> None:
    net = _make_p2p_network(n_agents=3, shape=(2,))

    x0s = initial_states(None, net)

    assert set(x0s) == set(net.graph)
    for agent in net.graph:
        np.testing.assert_array_equal(iop.to_numpy(x0s[agent]), np.zeros(agent.cost.shape))


def test_initial_states_array_broadcasts_to_all_agents() -> None:
    net = _make_p2p_network(n_agents=3, shape=(2,))
    shared_x0 = Array(np.array([1.5, -2.5]))

    x0s = initial_states(shared_x0, net)

    for agent in net.graph:
        np.testing.assert_array_equal(iop.to_numpy(x0s[agent]), np.array([1.5, -2.5]))


def test_initial_states_dict_matches_by_agent_id() -> None:
    net = _make_p2p_network(n_agents=3, shape=(2,))
    x0_dict: dict[Agent, np.ndarray] = {}

    for agent in net.graph:
        same_id_other_instance = Agent(agent.id, L2RegularizerCost(agent.cost.shape))
        x0_dict[same_id_other_instance] = np.full(agent.cost.shape, fill_value=float(agent.id + 1))

    x0s = initial_states(x0_dict, net)

    for agent in net.graph:
        np.testing.assert_array_equal(iop.to_numpy(x0s[agent]), np.full(agent.cost.shape, float(agent.id + 1)))


def test_initial_states_non_fed_missing_agent_raises() -> None:
    net = _make_p2p_network(n_agents=3, shape=(2,))
    x0_dict = {agent: iop.zeros(shape=agent.cost.shape, framework=agent.cost.framework, device=agent.cost.device) for agent in list(net.graph)[:2]}

    with pytest.raises(ValueError, match="x0 not provided for agent"):
        initial_states(x0_dict, net)


def test_initial_states_invalid_key_type_raises() -> None:
    net = _make_p2p_network(n_agents=2, shape=(2,))
    x0_dict = {0: np.zeros((2,))}

    with pytest.raises(TypeError, match="must have keys of type Agent"):
        initial_states(x0_dict, net)


def test_initial_states_invalid_type_raises() -> None:
    net = _make_p2p_network(n_agents=2, shape=(2,))

    with pytest.raises(ValueError, match="Invalid x0"):
        initial_states([1, 2], net)


def test_initial_states_fed_infers_server_as_client_mean() -> None:
    clients = [Agent(0, L2RegularizerCost((2,))), Agent(1, L2RegularizerCost((2,)))]
    net = FedNetwork(clients=clients)

    x0_dict = {
        clients[0]: np.array([1.0, 3.0]),
        clients[1]: np.array([5.0, 7.0]),
    }

    x0s = initial_states(x0_dict, net)

    expected_server = np.array([3.0, 5.0])
    np.testing.assert_array_equal(iop.to_numpy(x0s[net.server()]), expected_server)
    for client in clients:
        assert client in x0s


def test_initial_states_fed_missing_client_raises() -> None:
    clients = [Agent(0, L2RegularizerCost((2,))), Agent(1, L2RegularizerCost((2,)))]
    net = FedNetwork(clients=clients)

    with pytest.raises(ValueError, match="x0 not provided for agent"):
        initial_states({clients[0]: np.array([1.0, 2.0])}, net)


def test_normal_initialization_returns_expected_shapes() -> None:
    net = _make_p2p_network(n_agents=4, shape=(3,))

    x0s = normal_initialization(net, mean=0.0, std=0.5)

    assert set(x0s) == set(net.graph)
    for agent in net.graph:
        assert iop.shape(x0s[agent]) == agent.cost.shape


def test_uniform_initialization_validates_bounds() -> None:
    net = _make_p2p_network(n_agents=2, shape=(2,))

    with pytest.raises(ValueError, match="Expected high > low"):
        uniform_initialization(net, low=1.0, high=1.0)


def test_uniform_initialization_samples_within_range() -> None:
    net = _make_p2p_network(n_agents=3, shape=(4,))

    x0s = uniform_initialization(net, low=-2.0, high=-1.0)

    for agent in net.graph:
        values = iop.to_numpy(x0s[agent])
        assert np.all(values >= -2.0)
        assert np.all(values < -1.0)


def test_infer_client_weight_uses_A_first() -> None:
    client = Agent(0, L2RegularizerCost((2,)))
    client.cost.A = np.zeros((7, 2))  # type: ignore[attr-defined]

    weight = infer_client_weight(client)

    assert weight == 7.0


def test_infer_client_weight_falls_back_to_b() -> None:
    class UnsupportedArray:
        pass

    client = Agent(0, L2RegularizerCost((2,)))
    client.cost.A = UnsupportedArray()  # type: ignore[attr-defined]
    client.cost.b = np.zeros((5,))  # type: ignore[attr-defined]

    weight = infer_client_weight(client)

    assert weight == 5.0


def test_infer_client_weight_falls_back_to_n_samples() -> None:
    class UnsupportedArray:
        pass

    client = Agent(0, L2RegularizerCost((2,)))
    client.cost.A = UnsupportedArray()  # type: ignore[attr-defined]
    client.cost.b = UnsupportedArray()  # type: ignore[attr-defined]
    client.cost.n_samples = 11  # type: ignore[attr-defined]

    weight = infer_client_weight(client)

    assert weight == 11.0


def test_infer_client_weight_raises_when_no_size_signal() -> None:
    client = Agent(0, L2RegularizerCost((2,)))

    with pytest.raises(ValueError, match="Cannot infer client data size"):
        infer_client_weight(client)


def test_pytorch_initialization_rejects_non_pytorch_cost() -> None:
    net = _make_p2p_network(n_agents=1, shape=(2,))

    with pytest.raises(TypeError, match="expected PyTorchCost"):
        pytorch_initialization(net)


def test_pytorch_initialization_extracts_flattened_model_parameters() -> None:
    torch = pytest.importorskip("torch")

    dataset = [
        (torch.tensor([1.0, 2.0]), torch.tensor([1.0])),
        (torch.tensor([2.0, 3.0]), torch.tensor([2.0])),
    ]
    model = SimpleLinearModel(input_size=2, hidden_sizes=[3], output_size=1)
    cost = PyTorchCost(dataset=dataset, model=model, loss_fn=torch.nn.MSELoss())
    agent = Agent(0, cost)
    net = P2PNetwork(graph=nx.complete_graph(1), agents=[agent])

    x0s = pytorch_initialization(net)

    expected = cost._get_model_parameters().detach().cpu().numpy()  # noqa: SLF001
    np.testing.assert_allclose(iop.to_numpy(x0s[agent]), expected)
    assert iop.shape(x0s[agent]) == cost.shape
