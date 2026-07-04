from collections.abc import Sequence

from decent_bench.datasets._dataset_handler import DatasetHandler

from ._utils import _partition_counts, _shuffled_range, _split_by_counts


def split_size(dataset: DatasetHandler, partition_sizes: Sequence[int]) -> list[list[int]]:
    """
    Randomly split dataset indices into explicitly sized partitions.

    Args:
        dataset: Dataset handler whose indices are partitioned.
        partition_sizes: Number of datapoints assigned to each partition.

    """
    if len(partition_sizes) == 0:
        raise ValueError("partition_sizes must contain at least one partition size")

    counts = _partition_counts(len(dataset), len(partition_sizes), partition_sizes)
    return _split_by_counts(_shuffled_range(len(dataset)), counts)
