from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TypeAlias, TypeVar

from numpy import float64
from numpy.typing import NDArray
from sklearn import datasets

from decent_bench.utils.types import TensorLike

_T = TypeVar("_T", bound=TensorLike)

DatasetPartition: TypeAlias = tuple[_T, _T]  # noqa: UP040
"""Tuple of (A, b) representing one dataset partition."""


class Dataset[T: TensorLike](ABC):
    """Dataset containing partitions in the form of feature matrix A and target vector b."""

    @abstractmethod
    def training_partitions(self) -> Sequence[DatasetPartition[T]]:
        """Partitions used for finding the optimal optimization variable x."""


class SyntheticClassificationData(Dataset[NDArray[float64]]):
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
        self, n_partitions: int, n_classes: int, n_samples_per_partition: int, n_features: int, seed: int | None = None
    ):
        self.n_partitions = n_partitions
        self.n_classes = n_classes
        self.n_samples_per_partition = n_samples_per_partition
        self.n_features = n_features
        self.seed = seed

    def training_partitions(self) -> list[DatasetPartition[NDArray[float64]]]:  # noqa: D102
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
            res.append(partition)
        return res
