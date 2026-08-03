from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

import decent_bench.utils.interoperability as iop
from decent_bench.datasets._dataset_handler import DatasetHandler


def _validate_positive_int(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_non_negative_int(value: int, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _validate_samples_per_partition(
    samples_per_partition: int | Sequence[int] | None,
    n_partitions: int,
) -> int | list[int] | None:
    if samples_per_partition is None:
        return None
    if isinstance(samples_per_partition, int):
        _validate_non_negative_int(samples_per_partition, "samples_per_partition")
        return samples_per_partition

    sizes = list(samples_per_partition)
    if len(sizes) != n_partitions:
        raise ValueError("samples_per_partition must have one size per partition")
    for size in sizes:
        _validate_non_negative_int(size, "samples_per_partition")
    return sizes


def _partition_counts(
    n_datapoints: int,
    n_partitions: int,
    samples_per_partition: int | Sequence[int] | None,
) -> list[int]:
    _validate_non_negative_int(n_datapoints, "n_datapoints")
    _validate_positive_int(n_partitions, "n_partitions")
    normalized_samples_per_partition = _validate_samples_per_partition(samples_per_partition, n_partitions)
    if normalized_samples_per_partition is None:
        counts = _even_counts(n_datapoints, n_partitions)
    elif isinstance(normalized_samples_per_partition, int):
        counts = [normalized_samples_per_partition] * n_partitions
    else:
        counts = normalized_samples_per_partition

    total = sum(counts)
    if total > n_datapoints:
        raise ValueError(f"Requested {total} datapoints but dataset only has {n_datapoints}")
    return counts


def _even_counts(n_datapoints: int, n_partitions: int) -> list[int]:
    quotient, remainder = divmod(n_datapoints, n_partitions)
    return [quotient + int(partition_index < remainder) for partition_index in range(n_partitions)]


def _proportional_counts(counts: Sequence[int], total: int) -> list[int]:
    available_total = sum(counts)
    if available_total == 0:
        return [0 for _ in counts]

    exact_counts = [total * count / available_total for count in counts]
    proportional_counts = [int(count) for count in exact_counts]
    remainder = total - sum(proportional_counts)
    order = sorted(
        range(len(counts)),
        key=lambda index: exact_counts[index] - proportional_counts[index],
        reverse=True,
    )

    for index in order:
        if remainder == 0:
            break
        if proportional_counts[index] < counts[index]:
            proportional_counts[index] += 1
            remainder -= 1

    return proportional_counts


def _split_by_counts(indices: Sequence[int], counts: Sequence[int]) -> list[list[int]]:
    partitions: list[list[int]] = []
    start = 0
    for count in counts:
        end = start + count
        partitions.append(list(indices[start:end]))
        start = end
    return partitions


def _extend_weighted_by_capacity(
    indices: Sequence[int],
    partitions: list[list[int]],
    remaining_counts: list[int],
    weights: NDArray[Any],
) -> None:
    rng = iop.rng_numpy()
    for index in indices:
        available_partitions = [partition_index for partition_index, count in enumerate(remaining_counts) if count > 0]
        if len(available_partitions) == 0:
            return

        available_weights = np.asarray([weights[partition_index] for partition_index in available_partitions])
        available_weight_sum = float(np.sum(available_weights))
        probabilities = (
            np.full(len(available_partitions), 1 / len(available_partitions))
            if available_weight_sum == 0
            else available_weights / available_weight_sum
        )

        selected_partition = int(rng.choice(np.asarray(available_partitions), p=probabilities))
        partitions[selected_partition].append(index)
        remaining_counts[selected_partition] -= 1


def _normalized_labels(dataset: DatasetHandler) -> list[Hashable]:
    labels = dataset.get_labels()
    if len(labels) != len(dataset):
        raise ValueError(f"Expected {len(dataset)} labels, got {len(labels)}")
    return [_normalize_label(label) for label in labels]


def _normalize_label(label: object) -> Hashable:
    label_array = np.asarray(label)
    if label_array.ndim == 0:
        value = label_array.item()
        if isinstance(value, Hashable):
            return value
        return repr(value)

    values = tuple(label_array.reshape(-1).tolist())
    if all(isinstance(value, Hashable) for value in values):
        return values
    return repr(values)


def _indices_by_label(labels: Sequence[Hashable]) -> dict[Hashable, list[int]]:
    label_to_indices: dict[Hashable, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        label_to_indices[label].append(index)
    return label_to_indices


def _shuffled_range(stop: int) -> list[int]:
    return [int(index) for index in iop.rng_numpy().permutation(stop)]


def _shuffled_sequence[T](values: Sequence[T]) -> list[T]:
    indices = iop.rng_numpy().permutation(len(values))
    return [values[int(index)] for index in indices]
