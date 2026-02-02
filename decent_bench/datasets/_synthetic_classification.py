from sklearn import datasets

import decent_bench.utils.interoperability as iop
from decent_bench.utils.types import DatasetPartition, SupportedDevices, SupportedFrameworks

from ._dataset import Dataset


class SyntheticClassificationData(Dataset):
    def __init__(
        self,
        n_partitions: int,
        n_targets: int,
        n_features: int,
        n_samples_per_partition: int,
        *,
        framework: SupportedFrameworks = SupportedFrameworks.NUMPY,
        device: SupportedDevices = SupportedDevices.CPU,
        seed: int | None = None,
    ):
        """
        Dataset with synthetic classification data.

        Args:
            n_partitions: Number of training partitions to generate, i.e. the length of the sequence returned by
                :meth:`get_partitions`
            n_targets: Number of target dimensions (i.e. number of classes)
            n_features: Number of feature dimensions
            n_samples_per_partition: Number of samples per partition
            framework: Framework of the returned arrays
            device: Device of the returned arrays
            seed: Seed used for random generation, set to a specific value for reproducible results

        """
        self._n_partitions = n_partitions
        self._n_targets = n_targets
        self._n_samples_per_partition = n_samples_per_partition
        self._n_features = n_features
        self.framework = framework
        self.device = device
        self.seed = seed
        self._partitions: list[DatasetPartition] | None = None

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

    def get_datapoints(self) -> DatasetPartition:
        return [sample for partition in self.get_partitions() for sample in partition]

    def get_partitions(self) -> list[DatasetPartition]:
        if self._partitions is None:
            res: list[DatasetPartition] = []
            for i in range(self.n_partitions):
                seed = self.seed + i if self.seed is not None else None
                partition = datasets.make_classification(
                    n_samples=self._n_samples_per_partition,
                    n_features=self.n_features,
                    n_redundant=0,
                    n_classes=self.n_targets,
                    random_state=seed,
                )
                A = partition[0]  # noqa: N806
                b = partition[1]

                # Convert to list of tuples, one per sample
                partition_data = [
                    (iop.to_array(A[j], self.framework, self.device), iop.to_array(b[j], self.framework, self.device))
                    for j in range(self._n_samples_per_partition)
                ]
                res.append(partition_data)
            self._partitions = res

        return self._partitions

    def __len__(self) -> int:
        return self.n_samples
