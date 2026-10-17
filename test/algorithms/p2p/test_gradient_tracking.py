import networkx as nx
import numpy as np
from decent_array import Array
from decent_array import interoperability as iop

from decent_bench.agents import Agent
from decent_bench.algorithms.p2p import ED, EXTRA, KGT, LED, NIDS, ATC_Tracking, AugDGM, SimpleGT, WangElia
from decent_bench.algorithms.p2p._p2p_algorithm import P2PAlgorithm
from decent_bench.costs import QuadraticCost
from decent_bench.networks import P2PNetwork

_STEP_SIZE = 0.1
_INITIAL = np.array([[2.0], [2.0]])
_OFFSETS = np.array([[-1.0], [1.0]])


def _run(algorithm: P2PAlgorithm, iterations: int) -> tuple[np.ndarray, np.ndarray]:
    costs = [QuadraticCost(A=Array(np.ones((1, 1))), b=Array(offset)) for offset in _OFFSETS]
    agents = [Agent(cost) for cost in costs]
    network = P2PNetwork(graph=nx.path_graph(len(agents)), agents=agents)
    weights = iop.to_numpy(network.weights)

    algorithm.run(network, iterations=iterations)
    output = np.stack([iop.to_numpy(agent.x) for agent in network.agents()])
    return weights, output


def _gradient(x: np.ndarray) -> np.ndarray:
    return x + _OFFSETS


def test_simple_gt_two_steps() -> None:
    """Check two SimpleGT updates, including its carried tracker."""
    weights, actual = _run(SimpleGT(step_size=_STEP_SIZE, x0=Array(_INITIAL[0])), 2)
    x = _INITIAL.copy()
    y = x.copy()
    for _ in range(2):
        y_new = x - _STEP_SIZE * _gradient(x)
        x, y = y_new - y + weights @ x, y_new

    np.testing.assert_allclose(actual, x)


def test_exact_diffusion_two_steps() -> None:
    """Check two exact-diffusion updates and its corrected weights."""
    weights, actual = _run(ED(step_size=_STEP_SIZE, x0=Array(_INITIAL[0])), 2)
    corrected_weights = 0.5 * (np.eye(len(_INITIAL)) + weights)
    x = _INITIAL.copy()
    y = x.copy()
    y_new = x - _STEP_SIZE * _gradient(x)
    for _ in range(2):
        x_new = corrected_weights @ (x + y_new - y)
        y, y_new, x = y_new, x_new - _STEP_SIZE * _gradient(x_new), x_new

    np.testing.assert_allclose(actual, x)


def test_extra_two_steps() -> None:
    """Check the initial and recurrent EXTRA updates."""
    weights, actual = _run(EXTRA(step_size=_STEP_SIZE, x0=Array(_INITIAL[0])), 2)
    corrected_weights = 0.5 * (np.eye(len(_INITIAL)) + weights)
    x0 = _INITIAL.copy()
    x1 = weights @ x0 - _STEP_SIZE * _gradient(x0)
    x2 = x1 + weights @ x1 - corrected_weights @ x0 - _STEP_SIZE * (_gradient(x1) - _gradient(x0))

    np.testing.assert_allclose(actual, x2)


def test_atc_tracking_two_steps() -> None:
    """Check ATC-Tracking updates to the state and gradient tracker."""
    weights, actual = _run(ATC_Tracking(step_size=_STEP_SIZE, x0=Array(_INITIAL[0])), 2)
    x = _INITIAL.copy()
    y = _gradient(x)
    for _ in range(2):
        x_new = weights @ (x - _STEP_SIZE * y)
        y_new = weights @ y + _gradient(x_new) - _gradient(x)
        x, y = x_new, y_new

    np.testing.assert_allclose(actual, x)


def test_aug_dgm_two_steps() -> None:
    """Check Aug-DGM updates to the state and gradient tracker."""
    weights, actual = _run(AugDGM(step_size=_STEP_SIZE, x0=Array(_INITIAL[0])), 2)
    x = _INITIAL.copy()
    y = _gradient(x)
    for _ in range(2):
        x_new = weights @ (x - _STEP_SIZE * y)
        y_new = weights @ (y + _gradient(x_new) - _gradient(x))
        x, y = x_new, y_new

    np.testing.assert_allclose(actual, x)


def test_nids_two_steps() -> None:
    """Check the NIDS initialization and recurrent update."""
    weights, actual = _run(NIDS(step_size=_STEP_SIZE, x0=Array(_INITIAL[0])), 2)
    corrected_weights = 0.5 * (np.eye(len(_INITIAL)) + weights)
    x1 = _INITIAL - _STEP_SIZE * _gradient(_INITIAL)
    x2 = corrected_weights @ (2 * x1 - _INITIAL - _STEP_SIZE * _gradient(x1) + _STEP_SIZE * _gradient(_INITIAL))

    np.testing.assert_allclose(actual, x2)


def test_wang_elia_two_steps() -> None:
    """Check Wang-Elia updates to the state and auxiliary variable."""
    weights, actual = _run(WangElia(step_size=_STEP_SIZE, x0=Array(_INITIAL[0])), 2)
    mixing = 0.5 * (np.eye(len(_INITIAL)) - weights)
    x = _INITIAL.copy()
    z = np.zeros_like(x)
    for _ in range(2):
        x_new = x - mixing @ (x + z) - _STEP_SIZE * _gradient(x)
        z_new = z + mixing @ x
        x, z = x_new, z_new

    np.testing.assert_allclose(actual, x)


def test_kgt_two_steps() -> None:
    """Check KGT local steps, tracker update, and mixing."""
    weights, actual = _run(
        KGT(num_local_steps=2, step_size=0.1, aux_step_size=0.2, x0=Array(_INITIAL[0])),
        2,
    )
    x = _INITIAL.copy()
    c = np.zeros_like(x)
    for _ in range(2):
        x_local = x.copy()
        for _ in range(2):
            x_local -= 0.1 * (_gradient(x_local) + c)
        z = (x - x_local) / (2 * 0.1)
        message = x - (2 * 0.2 * 0.1) * z
        c = c - z + weights @ z
        x = weights @ message

    np.testing.assert_allclose(actual, x)


def test_led_two_steps() -> None:
    """Check LED local primal steps and dual tracking update."""
    weights, actual = _run(
        LED(num_local_steps=2, step_size=0.1, aux_step_size=0.2, x0=Array(_INITIAL[0])),
        2,
    )
    x = _INITIAL.copy()
    y = np.zeros_like(x)
    for _ in range(2):
        phi = x.copy()
        for _ in range(2):
            phi -= 0.1 * _gradient(phi) + 0.2 * y
        x = weights @ phi
        y += phi - x

    np.testing.assert_allclose(actual, x)
