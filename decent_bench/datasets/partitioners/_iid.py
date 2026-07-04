from collections.abc import Sequence

from decent_bench.datasets._dataset_handler import DatasetHandler

from ._utils import _partition_counts, _shuffled_range, _split_by_counts


def split_iid(
    dataset: DatasetHandler,
    n_partitions: int,
    samples_per_partition: int | Sequence[int] | None = None,
) -> list[list[int]]:
    """
    Randomly split dataset indices without replacement.

    The partitions are IID in expectation with respect to the dataset's empirical
    distribution. Labels are not inspected, so class balance is not guaranteed.

    Args:
        dataset: Dataset handler whose indices are partitioned.
        n_partitions: Number of partitions to create.
        samples_per_partition: Optional common partition size or one size per partition.

    """
    counts = _partition_counts(len(dataset), n_partitions, samples_per_partition)
    return _split_by_counts(_shuffled_range(len(dataset)), counts)
