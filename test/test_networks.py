import pytest
import networkx as nx
import numpy as np

from decent_bench.agents import Agent
from decent_bench.networks import P2PNetwork, FedNetwork
from decent_bench.costs import L2RegularizerCost
from decent_bench.utils import interoperability as iop
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks
from decent_bench.schemes import (
    AgentActivationScheme,
    AlwaysActive,
    CompressionScheme,
    DropScheme,
    NoiseScheme,
    NoCompression,
    NoDrops,
    NoNoise,
    UniformActivationRate,
)
from unittest.mock import MagicMock


class NeverActive(AgentActivationScheme):
    def is_active(self, iteration: int) -> bool:  # noqa: D102, ARG002
        return False


class MultiplyCompression(CompressionScheme):
    def __init__(self, factor: float):
        self.factor = factor

    def compress(self, msg):  # noqa: ANN001, D102
        return msg * self.factor


class AddNoise(NoiseScheme):
    def __init__(self, offset: float):
        self.offset = offset

    def make_noise(self, msg):  # noqa: ANN001, D102
        return msg + self.offset


class FixedDrop(DropScheme):
    def __init__(self, should_drop_message: bool):
        self.should_drop_message = should_drop_message

    def should_drop(self) -> bool:  # noqa: D102
        return self.should_drop_message


def test_p2p_network(n_agents: int = 10) -> None:
    net = P2PNetwork(
        graph=nx.complete_graph(n_agents),
        agents=[Agent(i, L2RegularizerCost((10,))) for i in range(n_agents)],
    )

    assert len(net.agents()) == n_agents

    check_degrees = [d == n_agents - 1 for d in net.degrees.values()]
    assert all(check_degrees) == True

    assert len(net.edges) == int(n_agents * (n_agents - 1) / 2)

    assert len(net.active_agents()) == n_agents

    x = iop.zeros(
        shape=net.agents()[0].cost.shape,
        framework=net.agents()[0].cost.framework,
        device=net.agents()[0].cost.device,
    )
    tot_msg = 0
    for i in net.agents():
        for j in net.neighbors(i):
            net.send(i, j, x)
            tot_msg += 1
    assert tot_msg == n_agents * (n_agents - 1)


def test_fed_network(n_agents: int = 10) -> None:
    net = FedNetwork(
        clients=[Agent(i, L2RegularizerCost((10,))) for i in range(n_agents)],
    )

    assert len(net.agents()) == n_agents

    assert len(net.active_agents()) == n_agents

    x = iop.zeros(
        shape=net.agents()[0].cost.shape,
        framework=net.agents()[0].cost.framework,
        device=net.agents()[0].cost.device,
    )
    tot_msg = 0
    for i in net.agents():
        net.send(i, msg=x)
        tot_msg += 1
    assert tot_msg == n_agents


def test_fed_network_accepts_custom_always_active_server() -> None:
    clients = [Agent(i, L2RegularizerCost((10,))) for i in range(3)]
    server = Agent(99, L2RegularizerCost((10,)), activation=AlwaysActive())

    net = FedNetwork(clients=clients, server=server)

    assert net.server() is server


def test_fed_network_rejects_custom_non_always_active_server() -> None:
    clients = [Agent(i, L2RegularizerCost((10,))) for i in range(3)]
    server = Agent(99, L2RegularizerCost((10,)), activation=UniformActivationRate(0.5))

    with pytest.raises(ValueError, match="FedNetwork server must use AlwaysActive activation"):
        FedNetwork(clients=clients, server=server)


def test_fed_network_default_server_is_always_active() -> None:
    clients = [Agent(i, L2RegularizerCost((10,))) for i in range(3)]

    net = FedNetwork(clients=clients)

    assert isinstance(net.server()._activation, AlwaysActive)  # noqa: SLF001


def test_p2p_network_rejects_mixed_framework_costs() -> None:
    agents = [
        Agent(0, L2RegularizerCost((2,), framework=SupportedFrameworks.NUMPY)),
        Agent(1, L2RegularizerCost((2,), framework=SupportedFrameworks.PYTORCH)),
    ]

    with pytest.raises(ValueError, match="same shape, framework, and device"):
        P2PNetwork(graph=nx.complete_graph(2), agents=agents)


def test_p2p_network_rejects_mixed_device_costs() -> None:
    agents = [
        Agent(0, L2RegularizerCost((2,), device=SupportedDevices.CPU)),
        Agent(1, L2RegularizerCost((2,), device=SupportedDevices.GPU)),
    ]

    with pytest.raises(ValueError, match="same shape, framework, and device"):
        P2PNetwork(graph=nx.complete_graph(2), agents=agents)


def test_fed_network_rejects_mixed_framework_clients() -> None:
    clients = [
        Agent(0, L2RegularizerCost((2,), framework=SupportedFrameworks.NUMPY)),
        Agent(1, L2RegularizerCost((2,), framework=SupportedFrameworks.PYTORCH)),
    ]

    with pytest.raises(ValueError, match="same shape, framework, and device"):
        FedNetwork(clients=clients)


def test_initialize_message_schemes_with_dict_all_agents() -> None:
    """Test that per-agent scheme dicts work when all agents are provided."""
    n_agents = 3
    agents = [Agent(i, L2RegularizerCost((10,))) for i in range(n_agents)]
    net = P2PNetwork(graph=nx.complete_graph(n_agents), agents=agents)

    # create per-agent noise schemes
    noise_schemes = {agent: NoNoise() for agent in agents}
    net._message_noise = net._initialize_message_schemes(noise_schemes, "noise", NoiseScheme)

    assert len(net._message_noise) == n_agents
    for agent in agents:
        assert agent in net._message_noise
        assert isinstance(net._message_noise[agent], NoNoise)


def test_initialize_message_schemes_with_dict_missing_agent() -> None:
    """Test that missing agent in scheme dict raises ValueError."""
    n_agents = 3
    agents = [Agent(i, L2RegularizerCost((10,))) for i in range(n_agents)]
    net = P2PNetwork(graph=nx.complete_graph(n_agents), agents=agents)

    # provide schemes for only 2 agents
    incomplete_schemes = {agents[0]: NoNoise(), agents[1]: NoNoise()}

    with pytest.raises(ValueError, match="scheme not provided for agent"):
        net._initialize_message_schemes(incomplete_schemes, "noise", NoiseScheme)


def test_initialize_message_schemes_with_dict_extra_keys() -> None:
    """Test that extra keys in scheme dict are ignored."""
    n_agents = 3
    agents = [Agent(i, L2RegularizerCost((10,))) for i in range(n_agents)]
    net = P2PNetwork(graph=nx.complete_graph(n_agents), agents=agents)

    # Provide schemes with extra agent not in network
    extra_agent = Agent(999, L2RegularizerCost((10,)))
    schemes_with_extra = {agent: NoNoise() for agent in agents}
    schemes_with_extra[extra_agent] = NoNoise()

    result = net._initialize_message_schemes(schemes_with_extra, "noise", NoiseScheme)

    assert len(result) == n_agents
    assert extra_agent not in result
    for agent in agents:
        assert agent in result


def test_initialize_message_schemes_dict_used_in_send() -> None:
    """Test that per-agent schemes in dict are actually used during send operations."""

    n_agents = 2
    agents = [Agent(i, L2RegularizerCost((10,))) for i in range(n_agents)]
    net = P2PNetwork(graph=nx.complete_graph(n_agents), agents=agents)

    # create mock compression schemes
    mock_schemes = {agents[0]: MagicMock(spec=CompressionScheme), agents[1]: MagicMock(spec=CompressionScheme)}
    mock_schemes[agents[0]].compress = MagicMock(side_effect=lambda x: x)
    mock_schemes[agents[1]].compress = MagicMock(side_effect=lambda x: x)

    net._message_compression = mock_schemes

    # mock drop and noise schemes to prevent actual operations
    net._message_drop = {agent: NoDrops() for agent in agents}
    net._message_noise = {agent: NoNoise() for agent in agents}

    msg = iop.zeros(shape=agents[0].cost.shape, framework=agents[0].cost.framework, device=agents[0].cost.device)
    net._send_one(agents[0], agents[1], msg)

    # verify agent 0's compression scheme was called
    mock_schemes[agents[0]].compress.assert_called_once()


def test_p2p_network_rejects_disconnected_graph() -> None:
    agents = [Agent(i, L2RegularizerCost((2,))) for i in range(3)]
    graph = nx.Graph()
    graph.add_edges_from([(0, 1)])
    graph.add_node(2)

    with pytest.raises(ValueError, match="graph needs to be connected"):
        P2PNetwork(graph=graph, agents=agents)


def test_p2p_network_rejects_directed_graph() -> None:
    agents = [Agent(i, L2RegularizerCost((2,))) for i in range(2)]
    graph = nx.DiGraph()
    graph.add_edge(0, 1)

    with pytest.raises(ValueError, match="Directed graphs are not supported"):
        P2PNetwork(graph=graph, agents=agents)


def test_p2p_network_rejects_multigraph() -> None:
    agents = [Agent(i, L2RegularizerCost((2,))) for i in range(2)]
    graph = nx.MultiGraph()
    graph.add_edge(0, 1)

    with pytest.raises(NotImplementedError, match="multi-graphs"):
        P2PNetwork(graph=graph, agents=agents)


def test_p2p_network_rejects_duplicate_agent_ids() -> None:
    agent_a = Agent(0, L2RegularizerCost((2,)))
    agent_b = Agent(0, L2RegularizerCost((2,)))
    graph = nx.Graph()
    graph.add_edge(agent_a, agent_b)

    with pytest.raises(ValueError, match="Agent IDs must be unique"):
        P2PNetwork(graph=graph)


def test_send_rejects_inactive_receiver() -> None:
    sender = Agent(0, L2RegularizerCost((2,)))
    inactive_receiver = Agent(1, L2RegularizerCost((2,)), activation=NeverActive())
    net = P2PNetwork(graph=nx.Graph([(sender, inactive_receiver)]))
    msg = iop.zeros(shape=(2,), framework=sender.cost.framework, device=sender.cost.device)

    with pytest.raises(ValueError, match="not active or not connected"):
        net.send(sender=sender, receiver=inactive_receiver, msg=msg)


@pytest.mark.parametrize(
    ("buffer_messages", "expect_message_after_step"),
    [
        (False, False),
        (True, True),
    ],
)
def test_step_clears_or_preserves_messages_based_on_buffer_setting(
    buffer_messages: bool, expect_message_after_step: bool
) -> None:
    sender = Agent(0, L2RegularizerCost((2,)))
    receiver = Agent(1, L2RegularizerCost((2,)))
    net = P2PNetwork(
        graph=nx.Graph([(sender, receiver)]),
        buffer_messages=buffer_messages,
    )
    msg = iop.to_array([1.0, -1.0], framework=sender.cost.framework, device=sender.cost.device)

    net.send(sender=sender, receiver=receiver, msg=msg)
    assert sender in receiver.messages

    net._step(1)  # noqa: SLF001

    assert (sender in receiver.messages) is expect_message_after_step


def test_send_applies_drop_compression_and_noise_schemes() -> None:
    sender = Agent(0, L2RegularizerCost((2,)))
    dropped_sender = Agent(1, L2RegularizerCost((2,)))
    receiver = Agent(2, L2RegularizerCost((2,)))
    net = P2PNetwork(
        graph=nx.complete_graph([sender, dropped_sender, receiver]),
        message_compression={
            sender: MultiplyCompression(2.0),
            dropped_sender: NoCompression(),
            receiver: NoCompression(),
        },
        message_noise={
            sender: AddNoise(3.0),
            dropped_sender: NoNoise(),
            receiver: NoNoise(),
        },
        message_drop={
            sender: FixedDrop(False),
            dropped_sender: FixedDrop(True),
            receiver: NoDrops(),
        },
    )
    msg = iop.to_array([1.0, 2.0], framework=sender.cost.framework, device=sender.cost.device)

    net.send(sender=sender, receiver=receiver, msg=msg)
    np_received = iop.to_numpy(receiver.messages[sender])
    np_expected = np.array([5.0, 7.0])
    assert np_received.shape == np_expected.shape
    assert np_received.tolist() == pytest.approx(np_expected.tolist())

    net.send(sender=dropped_sender, receiver=receiver, msg=msg)
    assert dropped_sender not in receiver.messages
