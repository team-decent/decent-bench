from collections.abc import Sequence
from dataclasses import dataclass
from functools import reduce
from operator import add
from typing import TYPE_CHECKING, Any

import networkx as nx
from networkx import Graph
from numpy import float64
from numpy.typing import NDArray

import decent_bench.centralized_algorithms as ca
from decent_bench.cost_functions import CostFunction, LinearRegressionCost, LogisticRegressionCost
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

if TYPE_CHECKING:
    AnyGraph = Graph[Any]
else:
    AnyGraph = Graph


@dataclass(eq=False)
class BenchmarkProblem:
    """
    Benchmark problem to run algorithms on, defining settings such as communication constraints and topology.

    Args:
        topology_structure: graph defining how agents are connected
        cost_functions: local cost functions, each one is given to one agent
        optimal_x: solution that minimizes the sum of the cost functions, used for calculating metrics
        agent_activation_schemes: setting for agent activation/participation, each scheme is applied to one agent
        compression_scheme: message compression setting
        noise_scheme: message noise setting
        drop_scheme: message drops setting

    """

    topology_structure: AnyGraph
    optimal_x: NDArray[float64]
    cost_functions: Sequence[CostFunction]
    agent_activation_schemes: Sequence[AgentActivationScheme]
    compression_scheme: CompressionScheme
    noise_scheme: NoiseScheme
    drop_scheme: DropScheme


def create_regression_problem(
    cost_function_cls: type[LinearRegressionCost | LogisticRegressionCost],
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
        cost_function_cls: type of cost function
        n_agents: number of agents
        n_neighbors_per_agent: number of neighbors per agent
        asynchrony: if true, agents only have a 50% probability of being active/participating at any given time
        compression: if true, messages are rounded to 4 significant digits
        noise: if true, messages are distorted by Gaussian noise
        drops: if true, messages have a 50% probability of being dropped

    """
    topology_structure = nx.random_regular_graph(n_neighbors_per_agent, n_agents, seed=0)
    dataset = SyntheticClassificationData(
        n_classes=2, n_partitions=n_agents, n_samples_per_partition=10, n_features=3, seed=0
    )
    costs = [cost_function_cls(*p) for p in dataset.training_partitions]
    sum_cost = reduce(add, costs)
    optimal_x = ca.accelerated_gradient_descent(sum_cost, x0=None, max_iter=50000, stop_tol=1e-100, max_tol=1e-16)
    agent_activation_schemes = [UniformActivationRate(0.5) if asynchrony else AlwaysActive()] * n_agents
    compression_scheme = Quantization(n_significant_digits=4) if compression else NoCompression()
    noise_scheme = GaussianNoise(mean=0, sd=0.001) if noise else NoNoise()
    drop_scheme = UniformDropRate(drop_rate=0.5) if drops else NoDrops()
    return BenchmarkProblem(
        topology_structure=topology_structure,
        cost_functions=costs,
        optimal_x=optimal_x,
        agent_activation_schemes=agent_activation_schemes,
        compression_scheme=compression_scheme,
        noise_scheme=noise_scheme,
        drop_scheme=drop_scheme,
    )
