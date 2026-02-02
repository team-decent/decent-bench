from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from decent_bench.utils.types import DatasetPartition


class Dataset(ABC):
    """
    Abstract wrapper for datasets used in distributed optimization benchmark problems.

    This class provides an interface for accessing datasets in a partitioned format
    for distributed optimization scenarios. Rather than storing the data directly,
    Dataset implementations act as wrappers that return data in the required format
    when queried.

    In distributed optimization, the dataset is typically divided among multiple
    agents in a network, where each agent has access to only a subset (partition)
    of the complete dataset. This class abstracts that partitioning scheme.

    When defining benchmark problems, a Dataset instance can be used to:

    - Provide local datasets to each agent in the network via :meth:`get_partitions`
    - Define the overall optimization problem (e.g., empirical risk minimization)
    - Serve as a test set for evaluating distributed algorithms on the full dataset
      (e.g. via :meth:`get_datapoints`) by assigning a :class:`~decent_bench.utils.types.DatasetPartition`
      to the test data of the benchmark problem.

    Data Structure:
        The dataset consists of datapoints, where each datapoint is a tuple of
        (features, target). Features and targets are represented as Array objects or
        framework-specific tensor objects in special cases. Partitions are sequences
        of such datapoints, allowing users to easily distribute local datasets among
        agents.

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
    def get_datapoints(self) -> DatasetPartition:
        """All datapoints in the dataset."""

    @abstractmethod
    def get_partitions(self) -> Sequence[DatasetPartition]:
        """
        Return the dataset divided into partitions for distribution among agents.

        This method provides the core partitioning functionality for distributed
        optimization. Each partition represents the local dataset of an agent in
        the network.

        Returns:
            Sequence of DatasetPartition objects, where each partition is a list of
            (features, target) tuples. The number of partitions typically corresponds
            to the number of agents in the network.

        """

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of datapoints in the dataset."""
