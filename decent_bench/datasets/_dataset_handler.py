from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from decent_bench.utils.types import Dataset


class DatasetHandler(ABC):
    """
    Abstract wrapper for datasets used in decentralized optimization benchmark problems.

    This class provides an interface for accessing datasets in a partitioned format
    for decentralized optimization scenarios. Rather than storing the data directly,
    :class:`DatasetHandler` implementations act as wrappers that return data in the required
    format when queried.

    In decentralized optimization, the dataset is typically divided among multiple
    agents in a network, where each agent has access to only a subset (partition)
    of the complete dataset. This class abstracts that partitioning scheme.

    When defining benchmark problems, a DatasetHandler instance can be used to:

    - Provide local datasets to each agent in the network via :meth:`get_partitions`
    - Define the overall optimization problem (e.g., empirical risk minimization)
    - Serve as a test set for evaluating decentralized algorithms on the full dataset
      (e.g. via :meth:`get_datapoints`) by assigning a :class:`~decent_bench.utils.types.Dataset`
      to the test data of the :class:`~decent_bench.benchmark_problem.BenchmarkProblem`.

    Data Structure:
        The dataset consists of datapoints, where each datapoint is a tuple of
        (features, targets). Features and targets are represented as :class:`~decent_bench.utils.array.Array`
        objects or framework-specific tensor objects in special cases. For unsupervised learning,
        targets are usually None. Partitions are sequences of such datapoints,
        allowing users to easily distribute local datasets among agents.

    Note:
        Implementations may load data from various sources (files, generators,
        synthetic data, etc) and are not required to store all datapoints in memory.

    """

    @property
    @abstractmethod
    def n_samples(self) -> int:
        """Total number of datapoints in the dataset."""

    @property
    @abstractmethod
    def n_partitions(self) -> int:
        """Total number of partitions in the dataset."""

    @property
    @abstractmethod
    def n_features(self) -> int:
        """Number of feature dimensions."""

    @property
    @abstractmethod
    def n_targets(self) -> int:
        """Number of target dimensions."""

    @abstractmethod
    def get_datapoints(self) -> Dataset:
        """
        Return all datapoints in the dataset.

        Can be used for evaluation on the full dataset or creation of test datasets.
        """

    @abstractmethod
    def get_partitions(self) -> Sequence[Dataset]:
        """
        Return the dataset divided into partitions for distribution among agents.

        This method provides the core partitioning functionality for decentralized
        optimization. Each partition represents the local dataset of an agent in
        the network.

        Returns:
            Sequence[Dataset]: Sequence of Dataset objects, where each partition is a list of
            (features, targets) tuples.

        """

    def __len__(self) -> int:
        """Return the number of datapoints in the dataset."""
        return self.n_samples
