from collections.abc import Sequence
from dataclasses import dataclass
from functools import reduce
from operator import add
from typing import TYPE_CHECKING, Any

import networkx as nx
from networkx import Graph

import decent_bench.centralized_algorithms as ca
from decent_bench.costs import Cost, LinearRegressionCost, LogisticRegressionCost
from decent_bench.datasets import SyntheticClassificationData
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
from decent_bench.utils.types import TensorLike

if TYPE_CHECKING:
    AnyGraph = Graph[Any]
    AnyCost = Cost[Any]
else:
    AnyGraph = Graph
    AnyCost = Cost


@dataclass(eq=False)
class BenchmarkProblem:
    """
    Benchmark problem to run algorithms on, defining settings such as communication constraints and topology.

    Args:
        network_structure: graph defining how agents are connected
        x_optimal: solution that minimizes the sum of the cost functions, used for calculating metrics
        costs: local cost functions, each one is given to one agent
        agent_activations: setting for agent activation/participation, each scheme is applied to one agent
        message_compression: message compression setting
        message_noise: message noise setting
        message_drop: message drops setting

    """

    network_structure: AnyGraph
    x_optimal: TensorLike
    costs: Sequence[AnyCost]
    agent_activations: Sequence[AgentActivationScheme]
    message_compression: CompressionScheme
    message_noise: NoiseScheme
    message_drop: DropScheme


def create_regression_problem(
    cost_cls: type[LinearRegressionCost | LogisticRegressionCost],
    *,
    n_agents: int = 100,
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
        n_neighbors_per_agent: number of neighbors per agent
        asynchrony: if true, agents only have a 50% probability of being active/participating at any given time
        compression: if true, messages are rounded to 4 significant digits
        noise: if true, messages are distorted by Gaussian noise
        drops: if true, messages have a 50% probability of being dropped

    """
    network_structure = nx.random_regular_graph(n_neighbors_per_agent, n_agents, seed=0)
    dataset = SyntheticClassificationData(
        n_classes=2, n_partitions=n_agents, n_samples_per_partition=10, n_features=3, seed=0
    )
    costs = [cost_cls(*p) for p in dataset.training_partitions()]
    sum_cost = reduce(add, costs)
    x_optimal = ca.accelerated_gradient_descent(sum_cost, x0=None, max_iter=50000, stop_tol=1e-100, max_tol=1e-16)
    agent_activations = [UniformActivationRate(0.5) if asynchrony else AlwaysActive()] * n_agents
    message_compression = Quantization(n_significant_digits=4) if compression else NoCompression()
    message_noise = GaussianNoise(mean=0, sd=0.001) if noise else NoNoise()
    message_drop = UniformDropRate(drop_rate=0.5) if drops else NoDrops()
    return BenchmarkProblem(
        network_structure=network_structure,
        costs=costs,
        x_optimal=x_optimal,
        agent_activations=agent_activations,
        message_compression=message_compression,
        message_noise=message_noise,
        message_drop=message_drop,
    )
