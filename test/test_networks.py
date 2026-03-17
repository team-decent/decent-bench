import pytest
import networkx as nx

from decent_bench.agents import Agent
from decent_bench.networks import P2PNetwork, FedNetwork
from decent_bench.costs import L2RegularizerCost
from decent_bench.utils import interoperability as iop


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
            net.receive(j, i)
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
        net.receive(net.server)
        tot_msg += 1
    assert tot_msg == n_agents