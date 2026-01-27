from sklearn import datasets

import decent_bench.utils.interoperability as iop
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks

from ._dataset import Dataset, DatasetPartition


class SyntheticClassificationData(Dataset):
    """
    Dataset with synthetic classification data.

    Args:
        n_partitions: number of training partitions to generate, i.e. the length of the sequence returned by
            :meth:`training_partitions`
        n_classes: number of classes, i.e. unique values in target vector b
        n_samples_per_partition: number of rows in A and b per partition
        n_features: columns in A
        framework: framework of the returned arrays
        device: device of the returned arrays
        seed: used for random generation, set to a specific value for reproducible results

    """

    def __init__(  # noqa: PLR0917
        self,
        n_partitions: int,
        n_classes: int,
        n_samples_per_partition: int,
        n_features: int,
        framework: SupportedFrameworks,
        device: SupportedDevices = SupportedDevices.CPU,
        seed: int | None = None,
    ):
        self.n_partitions = n_partitions
        self.n_classes = n_classes
        self.n_samples_per_partition = n_samples_per_partition
        self.n_features = n_features
        self.framework = framework
        self.device = device
        self.seed = seed
        self.res: list[DatasetPartition] | None = None

    def training_partitions(self) -> list[DatasetPartition]:
        if self.res is not None:
            return self.res

        res: list[DatasetPartition] = []
        for i in range(self.n_partitions):
            seed = self.seed + i if self.seed is not None else None
            partition = datasets.make_classification(
                n_samples=self.n_samples_per_partition,
                n_features=self.n_features,
                n_redundant=0,
                n_classes=self.n_classes,
                random_state=seed,
            )
            A = partition[0]  # noqa: N806
            b = partition[1]

            # Convert to list of tuples, one per sample
            partition_data = [
                (iop.to_array(A[j], self.framework, self.device), iop.to_array(b[j], self.framework, self.device))
                for j in range(self.n_samples_per_partition)
            ]
            res.append(partition_data)
        self.res = res
        return res
