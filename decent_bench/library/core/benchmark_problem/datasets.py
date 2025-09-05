from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import NewType

from numpy import float64
from numpy.typing import NDArray
from sklearn import datasets as sk_datasets

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


class SyntheticClassificationData(Dataset):
    """Dataset with synthetic classification data."""

    def __init__(
        self, partitions: int, classes: int, samples_per_partition: int, features: int, seed: int | None = None
    ):
        """
        Dataset with synthetic classification data.

        Args:
            classes: number of classes, i.e. unique values in the target vector b
            partitions: number of training partitions to generate, i.e. the length of the sequence returned by
                :attr:`training_partitions`
            samples_per_partition: number of rows in A and b per partition
            features: columns in A
            seed: used for random generation, set to a specific value for reproducible results

        """
        self.partitions = partitions
        self.classes = classes
        self.samples_per_partition = samples_per_partition
        self.features = features
        self.seed = seed

    @property
    def training_partitions(self) -> Sequence[DatasetPartition]:  # noqa: D102
        res = []
        for i in range(self.partitions):
            seed = self.seed + i if self.seed is not None else None
            partition = sk_datasets.make_classification(
                n_samples=self.samples_per_partition,
                n_features=self.features,
                n_redundant=0,
                n_classes=self.classes,
                random_state=seed,
            )
            res.append(DatasetPartition((A(partition[0]), b(partition[1]))))
        return res
