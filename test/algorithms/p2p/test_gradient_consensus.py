import networkx as nx
import numpy as np
from decent_array import Array
from decent_array import interoperability as iop

from decent_bench.agents import Agent
from decent_bench.algorithms.p2p import ATC, DGD
from decent_bench.costs import QuadraticCost
from decent_bench.networks import P2PNetwork

_STEP_SIZE = 0.1
_INITIAL = np.array([[2.0], [2.0]])
_OFFSETS = np.array([[-1.0], [1.0]])


def _make_network() -> tuple[P2PNetwork, np.ndarray]:
    costs = [QuadraticCost(A=Array(np.ones((1, 1))), b=Array(offset)) for offset in _OFFSETS]
    agents = [Agent(cost) for cost in costs]
    network = P2PNetwork(graph=nx.path_graph(len(agents)), agents=agents)
    return network, iop.to_numpy(network.weights)


def _run(algorithm: DGD | ATC, iterations: int) -> tuple[np.ndarray, np.ndarray]:
    network, weights = _make_network()
    algorithm.run(network, iterations=iterations)
    output = np.stack([iop.to_numpy(agent.x) for agent in network.agents()])
    return weights, output


def _gradient(x: np.ndarray) -> np.ndarray:
    """Compute gradients for the two scalar quadratic objectives."""
    return x + _OFFSETS


def test_dgd_update() -> None:
    """Check the DGD consensus and local gradient update."""
    weights, actual = _run(DGD(step_size=_STEP_SIZE, x0=Array(_INITIAL[0])), iterations=1)
    expected = weights @ _INITIAL - _STEP_SIZE * _gradient(_INITIAL)

    np.testing.assert_allclose(actual, expected)


def test_atc_update() -> None:
    """Check the ATC combination of locally adapted values."""
    weights, actual = _run(ATC(step_size=_STEP_SIZE, x0=Array(_INITIAL[0])), iterations=1)
    adapted = _INITIAL - _STEP_SIZE * _gradient(_INITIAL)

    np.testing.assert_allclose(actual, weights @ adapted)
