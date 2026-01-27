from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TypeAlias

from decent_bench.utils.array import Array

Datapoint: TypeAlias = tuple[Array, Array]  # noqa: UP040
"""Tuple of (x, y) representing one datapoint where x are features and y is the target."""

DatasetPartition: TypeAlias = list[Datapoint]  # noqa: UP040
"""List of datapoints representing one dataset partition."""


class Dataset(ABC):
    """Dataset containing partitions in the form of lists of tuples (feature, target)."""

    @abstractmethod
    def training_partitions(self) -> Sequence[DatasetPartition]:
        """Partitions used for finding the optimal optimization variable x."""
