from collections.abc import Hashable, Sequence

import numpy as np

import decent_bench.utils.interoperability as iop
from decent_bench.datasets._dataset_handler import DatasetHandler

from ._utils import (
    _extend_weighted_by_capacity,
    _indices_by_label,
    _normalized_labels,
    _partition_counts,
    _proportional_counts,
    _shuffled_sequence,
    _validate_non_negative_int,
    _validate_positive_int,
)


def split_dirichlet_label(
    dataset: DatasetHandler,
    n_partitions: int,
    alpha: float,
    *,
    samples_per_partition: int | Sequence[int] | None = None,
    min_partition_size: int = 0,
    max_retries: int = 100,
) -> list[list[int]]:
    """
    Split labels across partitions using a Dirichlet distribution.

    Args:
        dataset: Dataset handler whose indices are partitioned.
        n_partitions: Number of partitions to create.
        alpha: Dirichlet concentration. Smaller values create stronger label skew.
        samples_per_partition: Optional common partition size or one size per partition.
        min_partition_size: Minimum number of datapoints required in every partition.
        max_retries: Number of attempts to satisfy ``min_partition_size``.

    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    _validate_non_negative_int(min_partition_size, "min_partition_size")
    _validate_positive_int(max_retries, "max_retries")

    labels = _normalized_labels(dataset)
    partition_counts = _partition_counts(len(dataset), n_partitions, samples_per_partition)
    for _ in range(max_retries):
        partitions = _split_once(labels, partition_counts, alpha)
        if min(len(partition) for partition in partitions) >= min_partition_size:
            return partitions

    raise ValueError(
        f"Could not create {n_partitions} partitions with min_partition_size "
        f"{min_partition_size} after {max_retries} attempts"
    )


def _split_once(
    labels: Sequence[Hashable],
    partition_counts: Sequence[int],
    alpha: float,
) -> list[list[int]]:
    n_partitions = len(partition_counts)
    partitions: list[list[int]] = [[] for _ in range(n_partitions)]
    remaining_counts = list(partition_counts)
    label_items = list(_indices_by_label(labels).items())
    label_counts = _proportional_counts(
        [len(indices) for _, indices in label_items],
        sum(partition_counts),
    )

    label_assignments = _shuffled_sequence(list(zip(label_items, label_counts, strict=True)))
    for (_, label_indices), label_count in label_assignments:
        selected_indices = _shuffled_sequence(label_indices)[:label_count]
        proportions = iop.rng_numpy().dirichlet(np.full(n_partitions, alpha))
        _extend_weighted_by_capacity(selected_indices, partitions, remaining_counts, proportions)

    if any(remaining_counts):
        raise ValueError("Could not create Dirichlet partitions with the requested sizes")

    return [_shuffled_sequence(partition) for partition in partitions]
