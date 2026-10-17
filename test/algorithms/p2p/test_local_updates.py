import networkx as nx
import numpy as np
from decent_array import Array
from decent_array import interoperability as iop

from decent_bench.agents import Agent
from decent_bench.algorithms.p2p import ProxSkip
from decent_bench.costs import QuadraticCost
from decent_bench.networks import P2PNetwork


def test_prox_skip_when_communication_is_certain() -> None:
    """Check two ProxSkip updates with communication enabled each time."""
    step_size = 0.1
    aux_step_size = 0.2
    initial = np.array([[2.0], [2.0]])
    offsets = np.array([[-1.0], [1.0]])
    costs = [QuadraticCost(A=Array(np.ones((1, 1))), b=Array(offset)) for offset in offsets]
    agents = [Agent(cost) for cost in costs]
    network = P2PNetwork(graph=nx.path_graph(len(agents)), agents=agents)
    weights = iop.to_numpy(network.weights)

    ProxSkip(
        step_size=step_size,
        aux_step_size=aux_step_size,
        comm_probability=1.0,
        x0=Array(initial[0]),
    ).run(network, iterations=2)
    actual = np.stack([iop.to_numpy(agent.x) for agent in network.agents()])

    mixing = 0.5 * (np.eye(len(initial)) + weights)
    x = initial.copy()
    y = np.zeros_like(x)
    for _ in range(2):
        z = x - step_size * (x + offsets) - y
        x_new = mixing @ z
        y += aux_step_size * (z - x_new)
        x = x_new

    np.testing.assert_allclose(actual, x)
