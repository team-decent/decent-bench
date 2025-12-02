from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TypeAlias

from sklearn import datasets

import decent_bench.utils.interoperability as iop
from decent_bench.utils.parameter import X
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks

DatasetPartition: TypeAlias = tuple[X, X]  # noqa: UP040
"""Tuple of (A, b) representing one dataset partition."""


class Dataset(ABC):
    """Dataset containing partitions in the form of feature matrix A and target vector b."""

    @abstractmethod
    def training_partitions(self) -> Sequence[DatasetPartition]:
        """Partitions used for finding the optimal optimization variable x."""


class SyntheticClassificationData(Dataset):
    """
    Dataset with synthetic classification data.

    Args:
        n_partitions: number of training partitions to generate, i.e. the length of the sequence returned by
            :meth:`training_partitions`
        n_classes: number of classes, i.e. unique values in target vector b
        n_samples_per_partition: number of rows in A and b per partition
        n_features: columns in A
        seed: used for random generation, set to a specific value for reproducible results

    """

    def __init__(
        self,
        n_partitions: int,
        n_classes: int,
        n_samples_per_partition: int,
        n_features: int,
        framework: SupportedFrameworks,
        device: SupportedDevices,
        seed: int | None = None,
    ):
        self.n_partitions = n_partitions
        self.n_classes = n_classes
        self.n_samples_per_partition = n_samples_per_partition
        self.n_features = n_features
        self.framework = framework
        self.device = device
        self.seed = seed

    def training_partitions(self) -> list[DatasetPartition]:  # noqa: D102
        res = []
        for i in range(self.n_partitions):
            seed = self.seed + i if self.seed is not None else None
            partition = datasets.make_classification(
                n_samples=self.n_samples_per_partition,
                n_features=self.n_features,
                n_redundant=0,
                n_classes=self.n_classes,
                random_state=seed,
            )
            A = iop.numpy_to_X(partition[0], self.framework, self.device)  # noqa: N806
            b = iop.numpy_to_X(partition[1], self.framework, self.device)
            res.append((A, b))
        return res
