from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from decent_bench.benchmark_problem import BenchmarkProblem
from decent_bench.distributed_algorithms import Algorithm
from decent_bench.networks import Network


@dataclass
class BenchmarkResult:
    """
    Result of a benchmark execution, containing the results and metadata.

    This class is used to store the results and metadata of a benchmark execution.
    It is returned by the :func:`~decent_bench.benchmark.benchmark` function and contains
    all the information about the benchmark run, including the problem definition,
    algorithm states, table results, and plot results.

    * `problem`: contains the definition of the benchmark problem that was executed.
    * `states`: contains the final states of the algorithms after execution, organized by algorithm where
      each algorithm maps to a sequence of network states (one per trial).

    These results can be used to compute metrics after the benchmark run using
    :func:`~decent_bench.benchmark.compute_metrics`.
    """

    problem: BenchmarkProblem
    states: Mapping[Algorithm, Sequence[Network]]
