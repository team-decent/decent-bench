from collections.abc import Sequence
from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import DTypeLike, NDArray
from sklearn import datasets as sk_datasets

import decent_bench.utils.interoperability as iop
from decent_bench.utils.types import Dataset, SupportedDevices, SupportedFrameworks

from ._dataset_handler import DatasetHandler


class SyntheticClassificationDatasetHandler(DatasetHandler):
    def __init__(
        self,
        n_targets: int,
        n_features: int,
        n_samples: int,
        *,
        framework: SupportedFrameworks = SupportedFrameworks.NUMPY,
        device: SupportedDevices = SupportedDevices.CPU,
        feature_dtype: DTypeLike = np.float64,
        target_dtype: DTypeLike = np.int64,
        squeeze_targets: bool = False,
    ) -> None:
        """
        Dataset with synthetic classification data.

        Args:
            n_targets: Number of target dimensions (i.e. number of classes), returned as integers from 0 to n_targets-1
            n_features: Number of feature dimensions
            n_samples: Number of samples to generate before partitioning.
            framework: Framework of the returned arrays
            device: Device of the returned arrays
            feature_dtype: Data type of the features in the returned arrays
            target_dtype: Data type of the targets in the returned arrays
            squeeze_targets: If true, empty dimensions are removed from the targets, e.g. shape (1,) becomes ()

        """
        self._n_targets = n_targets
        self._n_samples = n_samples
        self._n_features = n_features
        self.framework = framework
        self.device = device
        self.feature_dtype = feature_dtype
        self.target_dtype = target_dtype
        self.squeeze_targets = squeeze_targets

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def n_features(self) -> int:
        return self._n_features

    @property
    def n_targets(self) -> int:
        return self._n_targets

    def get_datapoints(self) -> Dataset:
        features, targets = self._raw_data
        return self._create_partition(features, targets, list(range(self.n_samples)))

    def get_labels(self) -> list[int]:
        """Return generated classification labels."""
        return [int(label) for label in self._raw_data[1]]

    def split(
        self,
        partitions: Sequence[Sequence[int]],
    ) -> list[Dataset]:
        """Materialize generated samples from index partitions."""
        features, targets = self._raw_data
        idx_partitions = self._resolve_partitions(partitions)
        return [self._create_partition(features, targets, indices) for indices in idx_partitions]

    @cached_property
    def _raw_data(self) -> tuple[NDArray[Any], NDArray[Any]]:
        partition = sk_datasets.make_classification(
            n_samples=self._n_samples,
            n_features=self.n_features,
            n_redundant=0,
            n_classes=self.n_targets,
            random_state=iop.get_seed(),
        )
        features = partition[0].astype(self.feature_dtype)
        targets = partition[1].astype(self.target_dtype)
        return features, targets

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
