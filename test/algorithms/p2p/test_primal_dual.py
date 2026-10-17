import networkx as nx
import numpy as np
from decent_array import Array
from decent_array import interoperability as iop

from decent_bench.agents import Agent
from decent_bench.algorithms.p2p import ADMM, ATG, DLM, LT_ADMM, DiNNO
from decent_bench.costs import QuadraticCost
from decent_bench.networks import P2PNetwork

_INITIAL = np.array([[2.0], [2.0]])
_OFFSETS = np.array([[-1.0], [1.0]])


def _run(algorithm: ADMM | ATG | DLM | DiNNO | LT_ADMM, iterations: int) -> np.ndarray:
    costs = [QuadraticCost(A=Array(np.ones((1, 1))), b=Array(offset)) for offset in _OFFSETS]
    agents = [Agent(cost) for cost in costs]
    network = P2PNetwork(graph=nx.path_graph(len(agents)), agents=agents)
    algorithm.run(network, iterations=iterations)
    return np.stack([iop.to_numpy(agent.x) for agent in network.agents()])


def _gradient(x: np.ndarray) -> np.ndarray:
    return x + _OFFSETS


def _prox(x: np.ndarray, penalty: float) -> np.ndarray:
    return (x - penalty * _OFFSETS) / (1 + penalty)


def test_admm_two_updates() -> None:
    """Check ADMM proximal states and edge-variable updates."""
    penalty = 1.0
    relaxation = 0.5
    actual = _run(ADMM(penalty=penalty, relaxation=relaxation, z0=Array(np.zeros(1))), 2)
    z = np.zeros_like(_INITIAL)
    x = _INITIAL.copy()
    for _ in range(2):
        x = _prox(z / penalty, penalty)
        z = (1 - relaxation) * z - relaxation * (z - 2 * penalty * x[::-1])

    np.testing.assert_allclose(actual, x)


def test_dlm_three_iterations() -> None:
    """Check DLM's initial difference, primal updates, and dual state."""
    step_size = 0.1
    penalty = 0.5
    actual = _run(DLM(step_size=step_size, penalty=penalty, x0=Array(_INITIAL[0])), 3)
    x = _INITIAL.copy()
    difference = x - x[::-1]
    dual = np.zeros_like(x)
    for _ in range(2):
        x = x - step_size * (_gradient(x) + penalty * difference + dual)
        difference = x - x[::-1]
        dual += penalty * difference

    np.testing.assert_allclose(actual, x)


def test_atg_two_updates() -> None:
    """Check ATG's local primal and edge dual updates."""
    penalty = 0.5
    relaxation = 0.5
    gamma = 0.2
    delta = 0.1
    actual = _run(
        ATG(penalty=penalty, relaxation=relaxation, gamma=gamma, delta=delta, x0=Array(_INITIAL[0])),
        2,
    )
    x = _INITIAL.copy()
    z_y = np.zeros_like(x)
    z_s = np.zeros_like(x)
    for _ in range(2):
        y = (x + z_y) / (1 + penalty)
        s = (_gradient(x) + z_s) / (1 + penalty)
        x_new = (1 - gamma) * x + gamma * (y - delta * s)
        z_y, z_s = (
            (1 - relaxation) * z_y + relaxation * (-z_y[::-1] + 2 * penalty * y[::-1]),
            (1 - relaxation) * z_s + relaxation * (-z_s[::-1] + 2 * penalty * s[::-1]),
        )
        x = x_new

    np.testing.assert_allclose(actual, x)


def test_dinno_two_updates() -> None:
    """Check DiNNO dual ascent and its local augmented-gradient steps."""
    step_size = 0.05
    penalty = 0.5
    local_steps = 2
    actual = _run(
        DiNNO(step_size=step_size, penalty=penalty, num_local_steps=local_steps, x0=Array(_INITIAL[0])),
        2,
    )
    x = _INITIAL.copy()
    dual = np.zeros_like(x)
    for _ in range(2):
        dual += penalty * (x - x[::-1])
        psi = x.copy()
        for _ in range(local_steps):
            consensus_gradient = 2 * penalty * (psi - 0.5 * x - 0.5 * x[::-1])
            psi -= step_size * (_gradient(psi) + dual + consensus_gradient)
        x = psi

    np.testing.assert_allclose(actual, x)


def test_lt_admm_two_updates() -> None:
    """Check LT-ADMM local training and edge-variable updates."""
    step_size = 0.05
    aux_step_size = 0.1
    penalty = 0.5
    relaxation = 0.5
    local_steps = 2
    actual = _run(
        LT_ADMM(
            num_local_steps=local_steps,
            step_size=step_size,
            aux_step_size=aux_step_size,
            penalty=penalty,
            relaxation=relaxation,
            x0=Array(_INITIAL[0]),
        ),
        2,
    )
    x = _INITIAL.copy()
    z = _INITIAL.copy()
    for _ in range(2):
        correction = aux_step_size * (penalty * x - z)
        phi = x.copy()
        for _ in range(local_steps):
            phi -= step_size * _gradient(phi) + correction
        z = (1 - relaxation) * z - relaxation * (z - 2 * penalty * phi[::-1])
        x = phi

    np.testing.assert_allclose(actual, x)
