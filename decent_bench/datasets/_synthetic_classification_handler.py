import numpy as np
from numpy.typing import DTypeLike
from sklearn import datasets as sk_datasets

import decent_bench.utils.interoperability as iop
from decent_bench.utils.types import Dataset, SupportedDevices, SupportedFrameworks

from ._dataset_handler import DatasetHandler


class SyntheticClassificationDatasetHandler(DatasetHandler):
    def __init__(
        self,
        n_targets: int,
        n_features: int,
        n_samples_per_partition: int,
        n_partitions: int = 1,
        *,
        framework: SupportedFrameworks = SupportedFrameworks.NUMPY,
        device: SupportedDevices = SupportedDevices.CPU,
        feature_dtype: DTypeLike = np.float64,
        target_dtype: DTypeLike = np.int64,
        squeeze_targets: bool = False,
        seed: int | None = None,
    ):
        """
        Dataset with synthetic classification data.

        Args:
            n_partitions: Number of training partitions to generate, i.e. the length of the sequence returned by
                :meth:`get_partitions`
            n_targets: Number of target dimensions (i.e. number of classes), returned as integers from 0 to n_targets-1
            n_features: Number of feature dimensions
            n_samples_per_partition: Number of samples per partition
            framework: Framework of the returned arrays
            device: Device of the returned arrays
            feature_dtype: Data type of the features in the returned arrays
            target_dtype: Data type of the targets in the returned arrays
            squeeze_targets: If true, empty dimensions are removed from the targets, e.g. shape (1,) becomes ()
            seed: Seed used for random generation, set to a specific value for reproducible results

        """
        self._n_partitions = n_partitions
        self._n_targets = n_targets
        self._n_samples_per_partition = n_samples_per_partition
        self._n_features = n_features
        self.framework = framework
        self.device = device
        self.feature_dtype = feature_dtype
        self.target_dtype = target_dtype
        self.squeeze_targets = squeeze_targets
        self.seed = seed
        self._partitions: list[Dataset] | None = None

    @property
    def n_samples(self) -> int:
        return self.n_partitions * self._n_samples_per_partition

    @property
    def n_partitions(self) -> int:
        return self._n_partitions

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
            res: list[Dataset] = []
            for i in range(self.n_partitions):
                seed = self.seed + i if self.seed is not None else None
                partition = sk_datasets.make_classification(
                    n_samples=self._n_samples_per_partition,
                    n_features=self.n_features,
                    n_redundant=0,
                    n_classes=self.n_targets,
                    random_state=seed,
                )
                A = partition[0].astype(self.feature_dtype)  # noqa: N806
                b = partition[1].astype(self.target_dtype)

                # Convert to list of tuples, one per sample
                partition_data = [
                    (
                        iop.to_array(A[j], self.framework, self.device),
                        (
                            iop.squeeze(iop.to_array(b[j : j + 1], self.framework, self.device))
                            if self.squeeze_targets
                            else iop.to_array(b[j : j + 1], self.framework, self.device)
                        ),
                    )
                    for j in range(self._n_samples_per_partition)
                ]
                res.append(partition_data)
            self._partitions = res

        return self._partitions
