from dataclasses import dataclass

from decent_bench.networks import Network
from decent_bench.utils.array import Array
from decent_bench.utils.types import Dataset


@dataclass(eq=False)
class BenchmarkProblem:
    """
    Dataclass containing all benchmark data.

    Subclass it to add more benchmark data (e.g. validation data).

    Args:
        network: network of agents, each with a local cost function. This network represents the
                 initial state of the network over which algorithms are executed. Specifically,
                 algorithms are executed over *copies* of this network, and those copies are
                 stored in :class:`~decent_bench.benchmark.BenchmarkResult`. `BenchmarkProblem.network`
                 will never be modified, in order to preserve information on the initial state
        x_optimal: optional `Array` representing the optimal solution
        test_data: optional `Dataset` containing test data

    Example:
        >>> from dataclasses import dataclass
        >>> from decent_bench.benchmark import BenchmarkProblem
        >>> from decent_bench.utils.types import Dataset
        >>>
        >>> @dataclass(eq=False)
        ... class MyBenchmarkProblem(BenchmarkProblem):
        ...     validation_data: Dataset

    """

    network: Network
    x_optimal: Array | None = None
    test_data: Dataset | None = None
