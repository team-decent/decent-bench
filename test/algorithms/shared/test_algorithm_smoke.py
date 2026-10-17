import networkx as nx
import numpy as np
import pytest
from decent_array import interoperability as iop

from decent_bench.agents import Agent
from decent_bench.algorithms.federated import (
    FedAdagrad,
    FedAdam,
    FedAlgorithm,
    FedAvg,
    FedDyn,
    FedLT,
    FedNova,
    FedPD,
    FedProx,
    FedYogi,
    Scaffold,
)
from decent_bench.algorithms.p2p import (
    ADMM,
    ATC,
    ATG,
    DGD,
    DLM,
    ED,
    EXTRA,
    GT_SAGA,
    GT_SARAH,
    GT_VR,
    KGT,
    LED,
    LT_ADMM,
    LT_ADMM_VR,
    NIDS,
    ATC_Tracking,
    AugDGM,
    DiNNO,
    P2PAlgorithm,
    ProxSkip,
    SimpleGT,
    WangElia,
)
from decent_bench.benchmark import create_regression_problem
from decent_bench.costs import LinearRegressionCost, PyTorchCost
from decent_bench.networks import FedNetwork, P2PNetwork
from decent_bench.schemes import GaussianNoise, Quantization, UniformActivationRate, UniformDropRate

num_iterations = 25

all_p2p_algs = pytest.mark.parametrize(
    ("algorithm_cls", "kwargs"),
    [
        (DGD, {"step_size": 0.1}),
        (ATC, {"step_size": 0.1}),
        (SimpleGT, {"step_size": 0.1}),
        (ED, {"step_size": 0.1}),
        (AugDGM, {"step_size": 0.1}),
        (WangElia, {"step_size": 0.1}),
        (EXTRA, {"step_size": 0.1}),
        (ATC_Tracking, {"step_size": 0.1}),
        (NIDS, {"step_size": 0.1}),
        (ADMM, {"penalty": 1.0, "relaxation": 0.5}),
        (ATG, {"penalty": 1.0, "relaxation": 0.5}),
        (DLM, {"step_size": 0.1, "penalty": 1.0}),
        (DiNNO, {"step_size": 0.1, "num_local_steps": 5}),
        (GT_VR, {"step_size": 0.1, "snapshot_prob": 0.5}),
        (GT_SAGA, {"step_size": 0.1}),
        (GT_SARAH, {"step_size": 0.1, "num_local_steps": 5}),
        (KGT, {"step_size": 0.1, "num_local_steps": 5}),
        (LED, {"step_size": 0.1, "num_local_steps": 5}),
        (LT_ADMM, {"step_size": 0.1, "num_local_steps": 5}),
        (LT_ADMM_VR, {"step_size": 0.1, "num_local_steps": 5, "v2": False}),
        (LT_ADMM_VR, {"step_size": 0.1, "num_local_steps": 5, "v2": True}),
        (ProxSkip, {"step_size": 0.1, "comm_probability": 0.5}),
    ],
)

all_fed_algs = pytest.mark.parametrize(
    ("algorithm_cls", "kwargs"),
    [
        (FedAvg, {"step_size": 0.1}),
        (FedDyn, {"step_size": 0.1}),
        (FedLT, {"step_size": 0.1}),
        (FedProx, {"step_size": 0.1}),
        (FedAdagrad, {"step_size": 0.1}),
        (FedNova, {"step_size": 0.1}),
        (FedPD, {"step_size": 0.1}),
        (FedYogi, {"step_size": 0.1}),
        (FedAdam, {"step_size": 0.1}),
        (Scaffold, {"step_size": 0.1}),
    ],
)


def _create_p2p_network(impairments: bool, cost_cls: type) -> P2PNetwork:
    if cost_cls is PyTorchCost:
        torch = pytest.importorskip("torch")

    try:
        costs, _, _ = create_regression_problem(
            cost_cls=cost_cls,
            n_agents=4,
        )
    except Exception:
        costs, _, _ = create_regression_problem(
            cost_cls=cost_cls,
            n_agents=4,
        )
    agents = [
        Agent(
            cost,
            activation=UniformActivationRate(0.8) if impairments else None,
        )
        for cost in costs
    ]
    return P2PNetwork(
        graph=nx.complete_graph(len(agents)),
        agents=agents,
        message_compression=Quantization(quantization_step=1e-2) if impairments else None,
        message_noise=GaussianNoise(0.0, 0.01) if impairments else None,
        message_drop=UniformDropRate(0.1) if impairments else None,
    )


def _create_fed_network(impairments: bool, cost_cls: type) -> FedNetwork:
    if cost_cls is PyTorchCost:
        torch = pytest.importorskip("torch")

    try:
        costs, _, _ = create_regression_problem(
            cost_cls=cost_cls,
            n_agents=4,
        )
    except Exception:
        costs, _, _ = create_regression_problem(
            cost_cls=cost_cls,
            n_agents=4,
        )
    agents = [
        Agent(
            cost,
            activation=UniformActivationRate(0.8) if impairments else None,
        )
        for cost in costs
    ]
    return FedNetwork(
        clients=agents,
        message_compression=Quantization(quantization_step=1e-2) if impairments else None,
        message_noise=GaussianNoise(0.0, 0.01) if impairments else None,
        message_drop=UniformDropRate(0.1) if impairments else None,
    )


def _assert_network_outputs_are_valid(network: P2PNetwork | FedNetwork) -> None:
    """Check every client and server state after an algorithm run."""
    for agent in network.graph.nodes:
        assert agent.x is not None
        state = iop.to_numpy(agent.x)
        assert state.shape == agent.cost.shape
        assert np.isfinite(state).all()


@all_p2p_algs
def test_p2p_algorithm_instantiation(algorithm_cls: type, kwargs: dict[str, float | int]) -> None:
    algorithm = algorithm_cls(**kwargs)
    assert isinstance(algorithm.name, str)


@all_fed_algs
def test_fed_algorithm_instantiation(algorithm_cls: type, kwargs: dict[str, float | int]) -> None:
    algorithm = algorithm_cls(**kwargs)
    assert isinstance(algorithm.name, str)


@pytest.mark.parametrize(
    "impairments",
    [False, True],
)
@all_p2p_algs
def test_p2p_algorithm_execution(
    algorithm_cls: type[P2PAlgorithm],
    kwargs: dict[str, float | int],
    impairments: bool,
) -> None:
    algorithm = algorithm_cls(**kwargs)
    network = _create_p2p_network(impairments, LinearRegressionCost)

    # Just check that it runs without errors
    algorithm.run(network, num_iterations)
    _assert_network_outputs_are_valid(network)


@pytest.mark.parametrize(
    "impairments",
    [False, True],
)
@all_fed_algs
def test_fed_algorithm_execution(
    algorithm_cls: type[FedAlgorithm],
    kwargs: dict[str, float | int],
    impairments: bool,
) -> None:
    algorithm = algorithm_cls(**kwargs)
    network = _create_fed_network(impairments, LinearRegressionCost)

    # Just check that it runs without errors
    algorithm.run(network, num_iterations)
    _assert_network_outputs_are_valid(network)
