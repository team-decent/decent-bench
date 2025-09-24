from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import NewType

from numpy import float64
from numpy.typing import NDArray

A = NewType("A", NDArray[float64])
"""Feature matrix type."""

b = NewType("b", NDArray[float64])
"""Target vector type."""

DatasetPartition = NewType("DatasetPartition", tuple[A, b])
"""Tuple of (A, b) representing one dataset partition."""


class Dataset(ABC):
    """Dataset containing partitions in the form of feature matrix A and target vector b."""

    @property
    @abstractmethod
    def training_partitions(self) -> Sequence[DatasetPartition]:
        """Partitions used for finding the optimal optimization variable x."""
