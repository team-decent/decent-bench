import networkx as nx
import pytest

from decent_bench.agents import Agent
from decent_bench.algorithms.federated import FedAdagrad, FedAdam, FedAlgorithm, FedAvg, FedLT, FedProx, FedNova, FedYogi, Scaffold
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
from decent_bench.benchmark import create_classification_problem
from decent_bench.costs import LogisticRegressionCost, PyTorchCost
from decent_bench.networks import FedNetwork, P2PNetwork
from decent_bench.schemes import GaussianNoise, Quantization, UniformActivationRate, UniformDropRate

all_p2p_algs = pytest.mark.parametrize(
    ("algorithm_cls", "kwargs"),
    [
        (DGD, {"iterations": 10, "step_size": 0.1}),
        (ATC, {"iterations": 10, "step_size": 0.1}),
        (SimpleGT, {"iterations": 10, "step_size": 0.1}),
        (ED, {"iterations": 10, "step_size": 0.1}),
        (AugDGM, {"iterations": 10, "step_size": 0.1}),
        (WangElia, {"iterations": 10, "step_size": 0.1}),
        (EXTRA, {"iterations": 10, "step_size": 0.1}),
        (ATC_Tracking, {"iterations": 10, "step_size": 0.1}),
        (NIDS, {"iterations": 10, "step_size": 0.1}),
        (ADMM, {"iterations": 10, "rho": 1.0, "alpha": 0.5}),
        (ATG, {"iterations": 10, "rho": 1.0, "alpha": 0.5}),
        (DLM, {"iterations": 10, "step_size": 0.1, "penalty": 1.0}),
        (DiNNO, {"iterations": 10, "step_size": 0.1, "num_local_steps": 5}),
        (GT_VR, {"iterations": 10, "step_size": 0.1, "snapshot_prob": 0.5}),
        (GT_SAGA, {"iterations": 10, "step_size": 0.1}),
        (GT_SARAH, {"iterations": 10, "step_size": 0.1, "num_local_steps": 5}),
        (KGT, {"iterations": 10, "step_size": 0.1, "num_local_steps": 5}),
        (LED, {"iterations": 10, "step_size": 0.1, "num_local_steps": 5}),
        (LT_ADMM, {"iterations": 10, "step_size": 0.1, "num_local_steps": 5}),
        (LT_ADMM_VR, {"iterations": 10, "step_size": 0.1, "num_local_steps": 5, "v2": False}),
        (LT_ADMM_VR, {"iterations": 10, "step_size": 0.1, "num_local_steps": 5, "v2": True}),
        (ProxSkip, {"iterations": 10, "step_size": 0.1, "comm_probability": 0.5}),
    ],
)

all_fed_algs = pytest.mark.parametrize(
    ("algorithm_cls", "kwargs"),
    [
        (FedAvg, {"iterations": 10, "step_size": 0.1}),
        (FedLT, {"iterations": 10, "step_size": 0.1}),
        (FedProx, {"iterations": 10, "step_size": 0.1}),
        (FedAdagrad, {"iterations": 10, "step_size": 0.1}),
        (FedNova, {"iterations": 10, "step_size": 0.1}),
        (FedYogi, {"iterations": 10, "step_size": 0.1}),
        (FedAdam, {"iterations": 10, "step_size": 0.1}),
        (Scaffold, {"iterations": 10, "step_size": 0.1}),
    ],
)


def _create_p2p_network(impairments: bool, cost_cls: type) -> P2PNetwork:
    if cost_cls is PyTorchCost:
        torch = pytest.importorskip("torch")

    try:
        costs, _, _ = create_classification_problem(
            cost_cls=cost_cls,
            n_agents=4,
            show_progress=False,
        )
    except Exception:
        # Bad solver might fail, will be updated soon...
        costs, _, _ = create_classification_problem(
            cost_cls=cost_cls,
            n_agents=4,
            show_progress=False,
        )
    agents = [
        Agent(
            i,
            cost,
            activation=UniformActivationRate(0.8) if impairments else None,
        )
        for i, cost in enumerate(costs)
    ]
    return P2PNetwork(
        graph=nx.complete_graph(len(agents)),
        agents=agents,
        message_compression=Quantization(8) if impairments else None,
        message_noise=GaussianNoise(0.0, 0.01) if impairments else None,
        message_drop=UniformDropRate(0.1) if impairments else None,
    )


def _create_fed_network(impairments: bool, cost_cls: type) -> FedNetwork:
    if cost_cls is PyTorchCost:
        torch = pytest.importorskip("torch")

    try:
        costs, _, _ = create_classification_problem(
            cost_cls=cost_cls,
            n_agents=4,
            show_progress=False,
        )
    except Exception:
        # Bad solver might fail, will be updated soon...
        costs, _, _ = create_classification_problem(
            cost_cls=cost_cls,
            n_agents=4,
            show_progress=False,
        )
    agents = [
        Agent(
            i,
            cost,
            activation=UniformActivationRate(0.8) if impairments else None,
        )
        for i, cost in enumerate(costs)
    ]
    return FedNetwork(
        clients=agents,
        message_compression=Quantization(8) if impairments else None,
        message_noise=GaussianNoise(0.0, 0.01) if impairments else None,
        message_drop=UniformDropRate(0.1) if impairments else None,
    )


@all_p2p_algs
def test_p2p_algorithm_instantiation(algorithm_cls: type, kwargs: dict[str, float | int]) -> None:
    algorithm = algorithm_cls(**kwargs)
    assert algorithm.iterations == 10
    assert isinstance(algorithm.name, str)


@all_fed_algs
def test_fed_algorithm_instantiation(algorithm_cls: type, kwargs: dict[str, float | int]) -> None:
    algorithm = algorithm_cls(**kwargs)
    assert algorithm.iterations == 10
    assert isinstance(algorithm.name, str)


@pytest.mark.parametrize(
    "impairments",
    [False, True],
)
@pytest.mark.parametrize(
    "cost_cls",
    [LogisticRegressionCost, PyTorchCost],
)
@all_p2p_algs
def test_p2p_algorithm_execution(
    algorithm_cls: type[P2PAlgorithm],
    kwargs: dict[str, float | int],
    impairments: bool,
    cost_cls: type,
) -> None:
    algorithm = algorithm_cls(**kwargs)
    network = _create_p2p_network(impairments, cost_cls)

    # Just check that it runs without errors
    if cost_cls is PyTorchCost and algorithm_cls in {ADMM}:
        # Assert that it raises and error due to the need for proximal updates, which are not implemented for PyTorchCost
        with pytest.raises(NotImplementedError, match="Proximal operator is not implemented for PyTorchCost"):
            algorithm.run(network)
    else:
        algorithm.run(network)


@pytest.mark.parametrize(
    "impairments",
    [False, True],
)
@pytest.mark.parametrize(
    "cost_cls",
    [LogisticRegressionCost, PyTorchCost],
)
@all_fed_algs
def test_fed_algorithm_execution(
    algorithm_cls: type[FedAlgorithm],
    kwargs: dict[str, float | int],
    impairments: bool,
    cost_cls: type,
) -> None:
    algorithm = algorithm_cls(**kwargs)
    network = _create_fed_network(impairments, cost_cls)

    # Just check that it runs without errors
    algorithm.run(network)
