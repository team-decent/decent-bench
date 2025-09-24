from collections.abc import Sequence

from sklearn import datasets as sk_datasets

from decent_bench.datasets.dataset import A, Dataset, DatasetPartition, b


class SyntheticClassificationData(Dataset):
    """
    Dataset with synthetic classification data.

    Args:
        n_partitions: number of training partitions to generate, i.e. the length of the sequence returned by
            :attr:`training_partitions`
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

    @property
    def training_partitions(self) -> Sequence[DatasetPartition]:  # noqa: D102
        res = []
        for i in range(self.n_partitions):
            seed = self.seed + i if self.seed is not None else None
            partition = sk_datasets.make_classification(
                n_samples=self.n_samples_per_partition,
                n_features=self.n_features,
                n_redundant=0,
                n_classes=self.n_classes,
                random_state=seed,
            )
            res.append(DatasetPartition((A(partition[0]), b(partition[1]))))
        return res
