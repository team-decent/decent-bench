from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx

from decent_bench.agents import Agent

if TYPE_CHECKING:
    AgentGraph = nx.Graph[Agent]
else:
    AgentGraph = nx.Graph


@dataclass(eq=False)
class BenchmarkProblem:
    """
    Dataclass containing all benchmark data.

    Subclass it to add more benchmark data (e.g. optimal solution, test data).

    Args:
        network: network of agents (each with a local cost function)

    Example:
        >>> from dataclasses import dataclass
        >>> from decent_bench.benchmark import BenchmarkProblem
        >>> from decent_bench.utils.array import Array
        >>> from decent_bench.utils.types import Dataset
        >>>
        >>> @dataclass(eq=False)
        ... class MyBenchmarkProblem(BenchmarkProblem):
        ...     x_optimal: Array
        ...     test_data: Dataset

    """

    network: AgentGraph
