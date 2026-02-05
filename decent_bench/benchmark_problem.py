from collections.abc import Sequence
from dataclasses import dataclass
from functools import reduce
from operator import add
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np

import decent_bench.centralized_algorithms as ca
from decent_bench.costs import Cost, LinearRegressionCost, LogisticRegressionCost, PyTorchCost
from decent_bench.datasets import SyntheticClassificationDatasetHandler, SyntheticRegressionDatasetHandler
from decent_bench.schemes import (
    AgentActivationScheme,
    AlwaysActive,
    CompressionScheme,
    DropScheme,
    GaussianNoise,
    NoCompression,
    NoDrops,
    NoiseScheme,
    NoNoise,
    Quantization,
    UniformActivationRate,
    UniformDropRate,
)
from decent_bench.utils.array import Array
from decent_bench.utils.types import Dataset, SupportedDevices, SupportedFrameworks

if TYPE_CHECKING:
    AnyGraph = nx.Graph[Any]
else:
    AnyGraph = nx.Graph


@dataclass(eq=False)
class BenchmarkProblem:
    """
    Benchmark problem to run algorithms on, defining settings such as communication constraints and topology.

    Args:
        network_structure: graph defining how agents are connected
        x_optimal: solution that minimizes the sum of the cost functions, used for calculating metrics
        costs: local cost functions, each one is given to one agent
        agent_state_snapshot_period: period for recording agent state snapshots, used for plot metrics
        agent_activations: setting for agent activation/participation, each scheme is applied to one agent
        message_compression: message compression setting
        message_noise: message noise setting
        message_drop: message drops setting
        test_data: optional test dataset for evaluating generalization performance

    """

    network_structure: AnyGraph
    x_optimal: Array
    costs: Sequence[Cost]
    agent_state_snapshot_period: int
    agent_activations: Sequence[AgentActivationScheme]
    message_compression: CompressionScheme
    message_noise: NoiseScheme
    message_drop: DropScheme
    test_data: Dataset | None = None


def create_classification_problem(
    cost_cls: type[LogisticRegressionCost | PyTorchCost],
    *,
    n_agents: int = 100,
    agent_state_snapshot_period: int = 1,
    n_neighbors_per_agent: int = 3,
    asynchrony: bool = False,
    compression: bool = False,
    noise: bool = False,
    drops: bool = False,
) -> BenchmarkProblem:
    """
    Create out-of-the-box classification problems.

    Args:
        cost_cls: type of cost function
        n_agents: number of agents
        agent_state_snapshot_period: period for recording agent state snapshots, used for plot metrics
        n_neighbors_per_agent: number of neighbors per agent
        asynchrony: if true, agents only have a 50% probability of being active/participating at any given time
        compression: if true, messages are rounded to 4 significant digits
        noise: if true, messages are distorted by Gaussian noise
        drops: if true, messages have a 50% probability of being dropped

    Raises:
        ValueError: if an unsupported cost class is provided
        ImportError: if PyTorchCost is selected but PyTorch is not installed

    """
    network_structure = nx.random_regular_graph(n_neighbors_per_agent, n_agents, seed=0)
    if cost_cls is PyTorchCost:
        try:
            import torch  # noqa: PLC0415
        except ImportError as e:
            raise ImportError("PyTorch must be installed to use PyTorchCost") from e

        from decent_bench.utils.pytorch_utils import ArgmaxActivation, SimpleLinearModel  # noqa: PLC0415

        def model_gen() -> torch.nn.Module:
            return SimpleLinearModel(
                input_size=3,
                hidden_sizes=[],
                activation=None,
                output_size=2,
            )

        dataset = SyntheticClassificationDatasetHandler(
            n_targets=2,
            n_partitions=n_agents,
            n_samples_per_partition=10,
            n_features=3,
            framework=SupportedFrameworks.PYTORCH,
            device=SupportedDevices.CPU,
            feature_dtype=np.float32,
            target_dtype=np.int64,
            squeeze_targets=True,
            seed=0,
        )
        # Mypy cannot infer that cost_cls is PyTorchCost here
        costs = [
            cost_cls(p, model_gen(), torch.nn.CrossEntropyLoss(), final_activation=ArgmaxActivation())  # type: ignore[call-arg, arg-type]
            for p in dataset.get_partitions()
        ]
        x_optimal = ca.pytorch_gradient_descent(costs, lr=0.01, epochs=10000, conv_tol=1e-6)  # type: ignore[arg-type]
    elif cost_cls is LogisticRegressionCost:
        dataset = SyntheticClassificationDatasetHandler(
            n_targets=2,
            n_partitions=n_agents,
            n_samples_per_partition=10,
            n_features=3,
            framework=SupportedFrameworks.NUMPY,
            device=SupportedDevices.CPU,
            seed=0,
        )
        costs = [cost_cls(p) for p in dataset.get_partitions()]  # type: ignore[call-arg]
        sum_cost = reduce(add, costs)
        x_optimal = ca.accelerated_gradient_descent(sum_cost, x0=None, max_iter=50000, stop_tol=1e-100, max_tol=1e-16)
    else:
        raise ValueError(f"Unsupported cost class: {cost_cls}")

    agent_activations = [UniformActivationRate(0.5) if asynchrony else AlwaysActive()] * n_agents
    message_compression = Quantization(n_significant_digits=4) if compression else NoCompression()
    message_noise = GaussianNoise(mean=0, sd=0.001) if noise else NoNoise()
    message_drop = UniformDropRate(drop_rate=0.5) if drops else NoDrops()
    return BenchmarkProblem(
        network_structure=network_structure,
        costs=costs,
        agent_state_snapshot_period=agent_state_snapshot_period,
        x_optimal=x_optimal,
        agent_activations=agent_activations,
        message_compression=message_compression,
        message_noise=message_noise,
        message_drop=message_drop,
        test_data=None,
    )


def create_regression_problem(
    cost_cls: type[LinearRegressionCost | PyTorchCost],
    *,
    n_agents: int = 100,
    agent_state_snapshot_period: int = 1,
    n_neighbors_per_agent: int = 3,
    asynchrony: bool = False,
    compression: bool = False,
    noise: bool = False,
    drops: bool = False,
) -> BenchmarkProblem:
    """
    Create out-of-the-box regression problems.

    Args:
        cost_cls: type of cost function
        n_agents: number of agents
        agent_state_snapshot_period: period for recording agent state snapshots, used for plot metrics
        n_neighbors_per_agent: number of neighbors per agent
        asynchrony: if true, agents only have a 50% probability of being active/participating at any given time
        compression: if true, messages are rounded to 4 significant digits
        noise: if true, messages are distorted by Gaussian noise
        drops: if true, messages have a 50% probability of being dropped

    Raises:
        ValueError: if an unsupported cost class is provided
        ImportError: if PyTorchCost is selected but PyTorch is not installed

    """
    network_structure = nx.random_regular_graph(n_neighbors_per_agent, n_agents, seed=0)
    if cost_cls is PyTorchCost:
        try:
            import torch  # noqa: PLC0415
        except ImportError as e:
            raise ImportError("PyTorch must be installed to use PyTorchCost") from e

        from decent_bench.utils.pytorch_utils import SimpleLinearModel  # noqa: PLC0415

        def model_gen() -> torch.nn.Module:
            return SimpleLinearModel(
                input_size=1,
                hidden_sizes=[],
                activation=None,
                output_size=1,
            )

        dataset = SyntheticRegressionDatasetHandler(
            n_targets=1,
            n_partitions=n_agents,
            n_samples_per_partition=10,
            n_features=1,
            framework=SupportedFrameworks.PYTORCH,
            device=SupportedDevices.CPU,
            feature_dtype=np.float32,
            target_dtype=np.float32,
            seed=0,
        )
        costs = [cost_cls(p, model_gen(), torch.nn.MSELoss()) for p in dataset.get_partitions()]  # type: ignore[call-arg, arg-type]
        x_optimal = ca.pytorch_gradient_descent(costs, lr=0.01, epochs=15000, conv_tol=1e-6)  # type: ignore[arg-type]
    elif cost_cls is LinearRegressionCost:
        dataset = SyntheticRegressionDatasetHandler(
            n_targets=1,
            n_partitions=n_agents,
            n_samples_per_partition=10,
            n_features=1,
            framework=SupportedFrameworks.NUMPY,
            device=SupportedDevices.CPU,
            seed=0,
        )
        costs = [cost_cls(p) for p in dataset.get_partitions()]  # type: ignore[call-arg]
        sum_cost = reduce(add, costs)
        x_optimal = ca.accelerated_gradient_descent(sum_cost, x0=None, max_iter=50000, stop_tol=1e-100, max_tol=1e-16)
    else:
        raise ValueError(f"Unsupported cost class: {cost_cls}")

    agent_activations = [UniformActivationRate(0.5) if asynchrony else AlwaysActive()] * n_agents
    message_compression = Quantization(n_significant_digits=4) if compression else NoCompression()
    message_noise = GaussianNoise(mean=0, sd=0.001) if noise else NoNoise()
    message_drop = UniformDropRate(drop_rate=0.5) if drops else NoDrops()
    return BenchmarkProblem(
        network_structure=network_structure,
        costs=costs,
        agent_state_snapshot_period=agent_state_snapshot_period,
        x_optimal=x_optimal,
        agent_activations=agent_activations,
        message_compression=message_compression,
        message_noise=message_noise,
        message_drop=message_drop,
        test_data=None,
    )
