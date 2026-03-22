import pytest
import networkx as nx

from decent_bench.agents import Agent
from decent_bench.networks import P2PNetwork, FedNetwork
from decent_bench.costs import L2RegularizerCost
from decent_bench.utils import interoperability as iop
from decent_bench.schemes import NoiseScheme, NoNoise, CompressionScheme, NoDrops
from unittest.mock import MagicMock


def test_p2p_network(n_agents: int = 10) -> None:
    net = P2PNetwork(
        graph=nx.complete_graph(n_agents),
        agents=[Agent(i, L2RegularizerCost((10,))) for i in range(n_agents)]
    )

    assert len(net.agents()) == n_agents

    check_degrees = [d == n_agents-1 for d in net.degrees.values()]
    assert all(check_degrees) == True

    assert len(net.edges) == int(n_agents*(n_agents-1)/2)

    assert len(net.active_agents(0)) == n_agents

    x = iop.zeros(net.agents()[0].cost.shape, net.agents()[0].cost.framework, net.agents()[0].cost.device)
    tot_msg = 0
    for i in net.agents():
        for j in net.neighbors(i):
            net.send(i, j, x)
            tot_msg += 1
    assert tot_msg == n_agents*(n_agents-1)


def test_fed_network(n_agents: int = 10) -> None:
    net = FedNetwork(
        clients=[Agent(i, L2RegularizerCost((10,))) for i in range(n_agents)]
    )

    assert len(net.agents()) == n_agents

    assert len(net.active_agents(0)) == n_agents

    x = iop.zeros(net.agents()[0].cost.shape, net.agents()[0].cost.framework, net.agents()[0].cost.device)
    tot_msg = 0
    for i in net.agents():
        net.send(i, msg=x)
        tot_msg += 1
    assert tot_msg == n_agents


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
    
    msg = iop.zeros(agents[0].cost.shape, agents[0].cost.framework, agents[0].cost.device)
    net._send_one(agents[0], agents[1], msg)
    
    # verify agent 0's compression scheme was called
    mock_schemes[agents[0]].compress.assert_called_once()
