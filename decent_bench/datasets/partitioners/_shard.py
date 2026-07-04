import numpy as np

from decent_bench.datasets._dataset_handler import DatasetHandler

from ._utils import _normalized_labels, _shuffled_range, _shuffled_sequence, _validate_positive_int


def split_shard(
    dataset: DatasetHandler,
    n_partitions: int,
    shards_per_partition: int,
) -> list[list[int]]:
    """
    Sort datapoints by label, cut them into shards, and assign shards randomly.

    Args:
        dataset: Dataset handler whose indices are partitioned.
        n_partitions: Number of partitions to create.
        shards_per_partition: Number of label-sorted shards assigned to each partition.

    """
    _validate_positive_int(n_partitions, "n_partitions")
    _validate_positive_int(shards_per_partition, "shards_per_partition")
    labels = _normalized_labels(dataset)
    total_shards = n_partitions * shards_per_partition
    if total_shards > len(dataset):
        raise ValueError("n_partitions * shards_per_partition must be <= the number of datapoints")

    sorted_indices = sorted(range(len(dataset)), key=lambda index: repr(labels[index]))
    shards = [[int(index) for index in shard] for shard in np.array_split(sorted_indices, total_shards)]
    shard_order = _shuffled_range(total_shards)

    partitions: list[list[int]] = []
    for partition_index in range(n_partitions):
        start = partition_index * shards_per_partition
        assigned_shards = shard_order[start : start + shards_per_partition]
        partition = [index for shard_index in assigned_shards for index in shards[shard_index]]
        partitions.append(_shuffled_sequence(partition))
    return partitions
