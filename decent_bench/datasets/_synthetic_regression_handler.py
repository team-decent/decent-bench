from typing import Any

import numpy as np
from numpy.typing import DTypeLike, NDArray
from sklearn import datasets as sk_datasets

import decent_bench.utils.interoperability as iop
from decent_bench.utils.types import Dataset, SupportedDevices, SupportedFrameworks

from ._dataset_handler import DatasetHandler
from ._partitioners import IidPartitioner, Partitioner


class SyntheticRegressionDatasetHandler(DatasetHandler):
    def __init__(
        self,
        n_targets: int,
        n_features: int,
        n_samples_per_partition: int,
        n_partitions: int | None = None,
        *,
        framework: SupportedFrameworks = SupportedFrameworks.NUMPY,
        device: SupportedDevices = SupportedDevices.CPU,
        feature_dtype: DTypeLike = np.float64,
        target_dtype: DTypeLike = np.float64,
        squeeze_targets: bool = False,
        partitioner: Partitioner | None = None,
    ):
        """
        Dataset with synthetic regression data.

        Args:
            n_partitions: Number of partitions for the default IID partitioner. Do not use together with partitioner.
            n_targets: Number of target dimensions
            n_features: Number of feature dimensions
            n_samples_per_partition: Number of samples per default partition, or the generated pool size multiplier when
                using a custom partitioner.
            framework: Framework of the returned arrays
            device: Device of the returned arrays
            feature_dtype: Data type of the features in the returned arrays
            target_dtype: Data type of the targets in the returned arrays
            squeeze_targets: If true, empty dimensions are removed from the targets, e.g. shape (1,) becomes ()
            partitioner: Optional partitioner defining the split of one generated global dataset. If omitted, an IID
                partitioner is created from n_partitions and n_samples_per_partition.

        Raises:
            ValueError: If partitioner is combined with n_partitions.

        """
        if partitioner is not None and n_partitions is not None:
            raise ValueError("partitioner cannot be combined with n_partitions")
        if partitioner is None:
            n_partitions = 1 if n_partitions is None else n_partitions
            partitioner = IidPartitioner(n_partitions=n_partitions, samples_per_partition=n_samples_per_partition)

        self._n_targets = n_targets
        self._n_samples_per_partition = n_samples_per_partition
        self._n_features = n_features
        self.framework = framework
        self.device = device
        self.feature_dtype = feature_dtype
        self.target_dtype = target_dtype
        self.squeeze_targets = squeeze_targets
        self.partitioner = partitioner
        self._partitions: list[Dataset] | None = None

    @property
    def n_samples(self) -> int:
        return len(self.get_datapoints())

    @property
    def n_partitions(self) -> int:
        return self.partitioner.n_partitions

    @property
    def n_features(self) -> int:
        return self._n_features

    @property
    def n_targets(self) -> int:
        return self._n_targets

    def get_datapoints(self) -> Dataset:
        return [sample for partition in self.get_partitions() for sample in partition]

    def get_partitions(self) -> list[Dataset]:
        if self._partitions is None:
            self._partitions = self._partitioner_split()

        return self._partitions

    def _partitioner_split(self) -> list[Dataset]:
        n_samples = self.n_partitions * self._n_samples_per_partition
        partition = sk_datasets.make_regression(
            n_samples=n_samples,
            n_features=self.n_features,
            n_informative=self.n_features,
            n_targets=self.n_targets,
            random_state=iop.get_seed(),
            tail_strength=0.0,
        )
        A = partition[0].astype(self.feature_dtype)  # noqa: N806
        b = partition[1].astype(self.target_dtype)
        labels = [b[j] for j in range(n_samples)] if self.partitioner.requires_labels else None
        idx_partitions = self.partitioner.partition(n_samples, labels=labels)
        return [self._create_partition(A, b, indices) for indices in idx_partitions]

    def _create_partition(self, features: NDArray[Any], targets: NDArray[Any], indices: list[int]) -> Dataset:
        return [
            (
                iop.to_array(features[j], self.framework, self.device),
                (
                    iop.squeeze(iop.to_array(targets[j : j + 1], self.framework, self.device))
                    if self.squeeze_targets
                    else iop.to_array(targets[j : j + 1], self.framework, self.device)
                ),
            )
            for j in indices
        ]
