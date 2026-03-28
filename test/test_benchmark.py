import logging
from typing import Any

import networkx as nx
import pytest

import decent_bench.utils.interoperability as iop
from decent_bench.agents import Agent
from decent_bench.benchmark import (
    BenchmarkProblem,
    benchmark,
    create_classification_problem,
)
from decent_bench.costs import LogisticRegressionCost, PyTorchCost
from decent_bench.distributed_algorithms import ADMM, ATC, DGD, Algorithm, FedAvg
from decent_bench.networks import FedNetwork, P2PNetwork
from decent_bench.schemes import GaussianNoise, Quantization, UniformActivationRate, UniformDropRate

try:
    import torch

    TORCH_AVAILABLE = True
    TORCH_CUDA_AVAILABLE = torch.cuda.is_available()
except ModuleNotFoundError:
    TORCH_AVAILABLE = False
    TORCH_CUDA_AVAILABLE = False

# Suppress JAX debug logs that cause issues during cleanup
logging.getLogger("jax").setLevel(logging.WARNING)


def _build_p2p_problem_and_algorithms(
    iterations: int,
    cost_cls: type[LogisticRegressionCost | PyTorchCost],
) -> tuple[BenchmarkProblem, list[Algorithm[Any]]]:
    # Keep n_agents low to avoid expensive optimization in tests.
    costs, x_optimal, test_data = create_classification_problem(
        cost_cls=cost_cls,
        n_agents=4,
    )
    agents = [Agent(i, cost, activation=UniformActivationRate(0.8)) for i, cost in enumerate(costs)]
    network = P2PNetwork(
        graph=nx.complete_graph(len(agents)),
        agents=agents,
        message_compression=Quantization(8),
        message_noise=GaussianNoise(0.0, 0.01),
        message_drop=UniformDropRate(0.1),
    )
    problem = BenchmarkProblem(network=network, x_optimal=x_optimal, test_data=test_data)
    algorithms: list[Algorithm[Any]] = [
        DGD(iterations=iterations, step_size=0.01),
        ATC(iterations=iterations, step_size=0.01),
    ] + (
        # ADMM does not work with PyTorchCost due to no Proximal
        [
            ADMM(iterations=iterations),
        ]
        if cost_cls is LogisticRegressionCost
        else []
    )
    return problem, algorithms


def _build_fed_problem_and_algorithms(
    iterations: int,
    cost_cls: type[LogisticRegressionCost | PyTorchCost],
) -> tuple[BenchmarkProblem, list[Algorithm[Any]]]:
    # Keep n_agents low to avoid expensive optimization in tests.
    costs, x_optimal, test_data = create_classification_problem(
        cost_cls=cost_cls,
        n_agents=4,
    )
    agents = [Agent(i, cost, activation=UniformActivationRate(0.8)) for i, cost in enumerate(costs)]
    network = FedNetwork(
        clients=agents,
        message_compression=Quantization(8),
        message_noise=GaussianNoise(0.0, 0.01),
        message_drop=UniformDropRate(0.1),
    )
    problem = BenchmarkProblem(network=network, x_optimal=x_optimal, test_data=test_data)
    algorithms: list[Algorithm[Any]] = [
        FedAvg(iterations=iterations, step_size=0.01),
    ]
    return problem, algorithms


@pytest.mark.parametrize(
    "cost_cls",
    [
        LogisticRegressionCost,
        pytest.param(
            PyTorchCost,
            marks=pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
        ),
    ],
)
@pytest.mark.filterwarnings(
    "ignore:os.fork\\(\\) was called.*:RuntimeWarning"
)  # Suppress warnings about fork in JAX during cleanup, causes the test to fail
def test_p2p(cost_cls: type[LogisticRegressionCost | PyTorchCost]) -> None:
    iop.set_seed(123)
    problem_5, algorithms_5 = _build_p2p_problem_and_algorithms(5, cost_cls=cost_cls)

    bench = benchmark(
        algorithms=algorithms_5,
        benchmark_problem=problem_5,
        n_trials=2,
    )

    assert bench is not None, "Expected bench to be created successfully"

    # Check that the agent states and history is correct
    for alg in bench.states:
        for trial_result in bench.states[alg]:
            assert len(trial_result.agents()) == 4
            for agent in trial_result.agents():
                assert len(agent._x_history) == 6, (
                    f"Expected 6 iterations for agent {agent.id}, got {len(agent._x_history)}"
                )  # 5 iterations + initial state


@pytest.mark.parametrize(
    "cost_cls",
    [
        LogisticRegressionCost,
        pytest.param(
            PyTorchCost,
            marks=pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
        ),
    ],
)
@pytest.mark.filterwarnings(
    "ignore:os.fork\\(\\) was called.*:RuntimeWarning"
)  # Suppress warnings about fork in JAX during cleanup, causes the test to fail
def test_fed(cost_cls: type[LogisticRegressionCost | PyTorchCost]) -> None:
    iop.set_seed(123)
    problem_5, algorithms_5 = _build_fed_problem_and_algorithms(5, cost_cls=cost_cls)

    bench = benchmark(
        algorithms=algorithms_5,
        benchmark_problem=problem_5,
        n_trials=2,
    )

    assert bench is not None, "Expected bench to be created successfully"

    # Check that the agent states and history is correct
    for alg in bench.states:
        for trial_result in bench.states[alg]:
            assert len(trial_result.agents()) == 4
            for agent in trial_result.agents():
                assert len(agent._x_history) == 6, (
                    f"Expected 6 iterations for agent {agent.id}, got {len(agent._x_history)}"
                )  # 5 iterations + initial state
