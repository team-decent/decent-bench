import networkx as nx
import pytest

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
        (DGD, {"iterations": num_iterations, "step_size": 0.1}),
        (ATC, {"iterations": num_iterations, "step_size": 0.1}),
        (SimpleGT, {"iterations": num_iterations, "step_size": 0.1}),
        (ED, {"iterations": num_iterations, "step_size": 0.1}),
        (AugDGM, {"iterations": num_iterations, "step_size": 0.1}),
        (WangElia, {"iterations": num_iterations, "step_size": 0.1}),
        (EXTRA, {"iterations": num_iterations, "step_size": 0.1}),
        (ATC_Tracking, {"iterations": num_iterations, "step_size": 0.1}),
        (NIDS, {"iterations": num_iterations, "step_size": 0.1}),
        (ADMM, {"iterations": num_iterations, "penalty": 1.0, "relaxation": 0.5}),
        (ATG, {"iterations": num_iterations, "penalty": 1.0, "relaxation": 0.5}),
        (DLM, {"iterations": num_iterations, "step_size": 0.1, "penalty": 1.0}),
        (DiNNO, {"iterations": num_iterations, "step_size": 0.1, "num_local_steps": 5}),
        (GT_VR, {"iterations": num_iterations, "step_size": 0.1, "snapshot_prob": 0.5}),
        (GT_SAGA, {"iterations": num_iterations, "step_size": 0.1}),
        (GT_SARAH, {"iterations": num_iterations, "step_size": 0.1, "num_local_steps": 5}),
        (KGT, {"iterations": num_iterations, "step_size": 0.1, "num_local_steps": 5}),
        (LED, {"iterations": num_iterations, "step_size": 0.1, "num_local_steps": 5}),
        (LT_ADMM, {"iterations": num_iterations, "step_size": 0.1, "num_local_steps": 5}),
        (LT_ADMM_VR, {"iterations": num_iterations, "step_size": 0.1, "num_local_steps": 5, "v2": False}),
        (LT_ADMM_VR, {"iterations": num_iterations, "step_size": 0.1, "num_local_steps": 5, "v2": True}),
        (ProxSkip, {"iterations": num_iterations, "step_size": 0.1, "comm_probability": 0.5}),
    ],
)

all_fed_algs = pytest.mark.parametrize(
    ("algorithm_cls", "kwargs"),
    [
        (FedAvg, {"iterations": num_iterations, "step_size": 0.1}),
        (FedDyn, {"iterations": num_iterations, "step_size": 0.1}),
        (FedLT, {"iterations": num_iterations, "step_size": 0.1}),
        (FedProx, {"iterations": num_iterations, "step_size": 0.1}),
        (FedAdagrad, {"iterations": num_iterations, "step_size": 0.1}),
        (FedNova, {"iterations": num_iterations, "step_size": 0.1}),
        (FedPD, {"iterations": num_iterations, "step_size": 0.1}),
        (FedYogi, {"iterations": num_iterations, "step_size": 0.1}),
        (FedAdam, {"iterations": num_iterations, "step_size": 0.1}),
        (Scaffold, {"iterations": num_iterations, "step_size": 0.1}),
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


@all_p2p_algs
def test_p2p_algorithm_instantiation(algorithm_cls: type, kwargs: dict[str, float | int]) -> None:
    algorithm = algorithm_cls(**kwargs)
    assert algorithm.iterations == num_iterations
    assert isinstance(algorithm.name, str)


@all_fed_algs
def test_fed_algorithm_instantiation(algorithm_cls: type, kwargs: dict[str, float | int]) -> None:
    algorithm = algorithm_cls(**kwargs)
    assert algorithm.iterations == num_iterations
    assert isinstance(algorithm.name, str)


@pytest.mark.parametrize(
    "impairments",
    [False, True],
)
@pytest.mark.parametrize(
    "cost_cls",
    [LinearRegressionCost, PyTorchCost],
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
    [LinearRegressionCost, PyTorchCost],
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
