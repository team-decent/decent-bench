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
        network: network of agents (each with a local cost function)
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
