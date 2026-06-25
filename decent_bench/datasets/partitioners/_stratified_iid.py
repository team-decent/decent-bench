from collections.abc import Sequence

from decent_bench.datasets._dataset_handler import DatasetHandler

from ._utils import (
    _indices_by_label,
    _normalized_labels,
    _partition_counts,
    _proportional_counts,
    _shuffled_sequence,
    _split_by_counts,
)


def split_stratified_iid(
    dataset: DatasetHandler,
    n_partitions: int,
    samples_per_partition: int | Sequence[int] | None = None,
) -> list[list[int]]:
    """
    Split every label proportionally across partitions.

    Unlike :func:`split_iid`, this function inspects labels and explicitly
    preserves their proportions as closely as integer partition sizes allow.

    Args:
        dataset: Dataset handler whose indices are partitioned.
        n_partitions: Number of partitions to create.
        samples_per_partition: Optional common partition size or one size per partition.

    """
    labels = _normalized_labels(dataset)
    partition_counts = _partition_counts(len(dataset), n_partitions, samples_per_partition)
    label_items = sorted(_indices_by_label(labels).items(), key=lambda item: repr(item[0]))
    label_counts = _proportional_counts(
        [len(indices) for _, indices in label_items],
        sum(partition_counts),
    )

    partitions: list[list[int]] = [[] for _ in range(n_partitions)]
    remaining_counts = partition_counts.copy()
    for (_, label_indices), label_count in zip(label_items, label_counts, strict=True):
        selected_indices = _shuffled_sequence(label_indices)[:label_count]
        counts = _proportional_counts(remaining_counts, label_count)
        for partition, indices in zip(partitions, _split_by_counts(selected_indices, counts), strict=True):
            partition.extend(indices)
        remaining_counts = [remaining - count for remaining, count in zip(remaining_counts, counts, strict=True)]

    if any(remaining_counts):
        raise ValueError("Could not create stratified partitions with the requested sizes")

    return [_shuffled_sequence(partition) for partition in partitions]
