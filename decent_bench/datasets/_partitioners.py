from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np

import decent_bench.utils.interoperability as iop


class Partitioner(ABC):
    """Base class for index-based dataset partitioners."""

    def __init__(self, n_partitions: int) -> None:
        _validate_positive_int(n_partitions, "n_partitions")
        self._n_partitions = n_partitions

    @property
    def n_partitions(self) -> int:
        """Number of partitions created by this partitioner."""
        return self._n_partitions

    @property
    def requires_labels(self) -> bool:
        """Whether this partitioner requires one label per datapoint."""
        return False

    @abstractmethod
    def partition(self, n_samples: int, labels: Sequence[Any] | None = None) -> list[list[int]]:
        """Return index partitions for a dataset with ``n_samples`` datapoints."""


class IidPartitioner(Partitioner):
    """Randomly split datapoints into IID partitions."""

    def __init__(self, n_partitions: int, samples_per_partition: int | None = None) -> None:
        """
        Initialize an IID partitioner.

        Args:
            n_partitions: Number of partitions to create.
            samples_per_partition: Optional fixed number of samples in every partition.

        """
        super().__init__(n_partitions)
        if samples_per_partition is not None:
            _validate_non_negative_int(samples_per_partition, "samples_per_partition")
        self.samples_per_partition = samples_per_partition

    def partition(self, n_samples: int, labels: Sequence[Any] | None = None) -> list[list[int]]:
        """Return random partitions sampled without replacement."""
        del labels

        _validate_non_negative_int(n_samples, "n_samples")
        counts = (
            _even_counts(n_samples, self.n_partitions)
            if self.samples_per_partition is None
            else [self.samples_per_partition] * self.n_partitions
        )
        _validate_total_count(counts, n_samples)

        indices = _shuffled_range(n_samples)
        return _split_by_counts(indices, counts)


class SizePartitioner(Partitioner):
    """Randomly split datapoints into explicitly sized partitions."""

    def __init__(self, partition_sizes: Sequence[int]) -> None:
        """
        Initialize a quantity-skew partitioner.

        Args:
            partition_sizes: Number of samples assigned to each partition.

        Raises:
            ValueError: If no partition sizes are provided or any size is negative.

        """
        if len(partition_sizes) == 0:
            raise ValueError("partition_sizes must contain at least one partition size")

        sizes = list(partition_sizes)
        for size in sizes:
            _validate_non_negative_int(size, "partition_size")

        super().__init__(len(sizes))
        self.partition_sizes = sizes

    def partition(self, n_samples: int, labels: Sequence[Any] | None = None) -> list[list[int]]:
        """Return random partitions with the configured sizes."""
        del labels

        _validate_non_negative_int(n_samples, "n_samples")
        _validate_total_count(self.partition_sizes, n_samples)

        indices = _shuffled_range(n_samples)
        return _split_by_counts(indices, self.partition_sizes)


class DirichletLabelPartitioner(Partitioner):
    """Split every label across clients using a Dirichlet distribution."""

    def __init__(
        self,
        n_partitions: int,
        alpha: float,
        *,
        min_partition_size: int = 0,
        max_retries: int = 100,
    ) -> None:
        """
        Initialize a Dirichlet label-skew partitioner.

        Args:
            n_partitions: Number of partitions to create.
            alpha: Dirichlet concentration. Smaller values create stronger label skew.
            min_partition_size: Optional minimum number of samples required in every partition.
            max_retries: Number of attempts to satisfy ``min_partition_size``.

        Raises:
            ValueError: If alpha is not positive, min_partition_size is negative, or max_retries is not positive.

        """
        super().__init__(n_partitions)
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        _validate_non_negative_int(min_partition_size, "min_partition_size")
        _validate_positive_int(max_retries, "max_retries")

        self.alpha = alpha
        self.min_partition_size = min_partition_size
        self.max_retries = max_retries

    @property
    def requires_labels(self) -> bool:
        """Whether this partitioner requires one label per datapoint."""
        return True

    def partition(self, n_samples: int, labels: Sequence[Any] | None = None) -> list[list[int]]:
        """
        Return label-skewed partitions sampled without replacement.

        Raises:
            ValueError: If labels are missing or the minimum partition size cannot be satisfied.

        """
        normalized_labels = _normalize_labels(n_samples, labels)
        for _ in range(self.max_retries):
            partitions = self._partition_once(normalized_labels)
            if min(len(partition) for partition in partitions) >= self.min_partition_size:
                return partitions

        raise ValueError(
            f"Could not create {self.n_partitions} partitions with min_partition_size "
            f"{self.min_partition_size} after {self.max_retries} attempts"
        )

    def _partition_once(self, labels: Sequence[Hashable]) -> list[list[int]]:
        partitions: list[list[int]] = [[] for _ in range(self.n_partitions)]
        label_to_indices = _indices_by_label(labels)
        rng = iop.rng_numpy()

        for label_indices in label_to_indices.values():
            shuffled_indices = _shuffled_sequence(label_indices)
            proportions = rng.dirichlet(np.full(self.n_partitions, self.alpha))
            counts = [int(count) for count in rng.multinomial(len(shuffled_indices), proportions)]
            for partition, indices in zip(partitions, _split_by_counts(shuffled_indices, counts), strict=True):
                partition.extend(indices)

        return [_shuffled_sequence(partition) for partition in partitions]


class PathologicalLabelPartitioner(Partitioner):
    """Sort datapoints by label, cut them into shards, and assign shards to clients."""

    def __init__(
        self,
        n_partitions: int,
        shards_per_partition: int,
        *,
        n_shards: int | None = None,
    ) -> None:
        """
        Initialize a shard-based pathological label partitioner.

        Args:
            n_partitions: Number of partitions to create.
            shards_per_partition: Number of label-sorted shards assigned to each partition.
            n_shards: Optional total number of shards. Defaults to
                ``n_partitions * shards_per_partition``.

        Raises:
            ValueError: If the shard configuration is invalid.

        """
        super().__init__(n_partitions)
        _validate_positive_int(shards_per_partition, "shards_per_partition")
        self.shards_per_partition = shards_per_partition
        self.n_shards = n_shards if n_shards is not None else n_partitions * shards_per_partition
        if self.n_shards != n_partitions * shards_per_partition:
            raise ValueError("n_shards must equal n_partitions * shards_per_partition")

    @property
    def requires_labels(self) -> bool:
        """Whether this partitioner requires one label per datapoint."""
        return True

    def partition(self, n_samples: int, labels: Sequence[Any] | None = None) -> list[list[int]]:
        """
        Return partitions made from randomly assigned label-sorted shards.

        Raises:
            ValueError: If labels are missing or more shards than samples are requested.

        """
        normalized_labels = _normalize_labels(n_samples, labels)
        if self.n_shards > n_samples:
            raise ValueError("n_shards must be <= n_samples")

        sorted_indices = sorted(range(n_samples), key=lambda index: repr(normalized_labels[index]))
        shards = [[int(index) for index in shard] for shard in np.array_split(sorted_indices, self.n_shards)]
        shard_order = _shuffled_range(self.n_shards)

        partitions: list[list[int]] = []
        for partition_index in range(self.n_partitions):
            start = partition_index * self.shards_per_partition
            assigned_shards = shard_order[start : start + self.shards_per_partition]
            partition = [index for shard_index in assigned_shards for index in shards[shard_index]]
            partitions.append(_shuffled_sequence(partition))

        return partitions


class ShardPartitioner(PathologicalLabelPartitioner):
    """Alias for :class:`PathologicalLabelPartitioner`."""


class LabelQuantityPartitioner(Partitioner):
    """Restrict each partition to a fixed number of labels."""

    def __init__(
        self,
        n_partitions: int,
        classes_per_partition: int,
        *,
        samples_per_partition: int | None = None,
    ) -> None:
        """
        Initialize a label-quantity partitioner.

        Args:
            n_partitions: Number of partitions to create.
            classes_per_partition: Number of classes allowed in each partition.
            samples_per_partition: Optional maximum number of samples returned per partition.

        """
        super().__init__(n_partitions)
        _validate_positive_int(classes_per_partition, "classes_per_partition")
        if samples_per_partition is not None:
            _validate_non_negative_int(samples_per_partition, "samples_per_partition")

        self.classes_per_partition = classes_per_partition
        self.samples_per_partition = samples_per_partition

    @property
    def requires_labels(self) -> bool:
        """Whether this partitioner requires one label per datapoint."""
        return True

    def partition(self, n_samples: int, labels: Sequence[Any] | None = None) -> list[list[int]]:
        """
        Return partitions where each client sees at most ``classes_per_partition`` labels.

        Raises:
            ValueError: If labels are missing or too many classes per partition are requested.

        """
        normalized_labels = _normalize_labels(n_samples, labels)
        label_to_indices = _indices_by_label(normalized_labels)
        sorted_labels = sorted(label_to_indices, key=repr)
        if self.classes_per_partition > len(sorted_labels):
            raise ValueError("classes_per_partition must be <= the number of unique labels")

        label_order = _shuffled_sequence(sorted_labels)
        client_label_groups = [
            {
                label_order[(partition_index * self.classes_per_partition + offset) % len(label_order)]
                for offset in range(self.classes_per_partition)
            }
            for partition_index in range(self.n_partitions)
        ]

        clients_by_label: dict[Hashable, list[int]] = defaultdict(list)
        for partition_index, label_group in enumerate(client_label_groups):
            for label in label_group:
                clients_by_label[label].append(partition_index)

        partitions: list[list[int]] = [[] for _ in range(self.n_partitions)]
        for label, indices in label_to_indices.items():
            assigned_clients = clients_by_label[label]
            if len(assigned_clients) == 0:
                continue

            shuffled_indices = _shuffled_sequence(indices)
            counts = _even_counts(len(shuffled_indices), len(assigned_clients))
            for partition_index, label_indices in zip(
                assigned_clients,
                _split_by_counts(shuffled_indices, counts),
                strict=True,
            ):
                partitions[partition_index].extend(label_indices)

        res = [_shuffled_sequence(partition) for partition in partitions]
        if self.samples_per_partition is not None:
            res = [partition[: self.samples_per_partition] for partition in res]

        return res


class ClassQuantityPartitioner(LabelQuantityPartitioner):
    """Alias for :class:`LabelQuantityPartitioner`."""


def _validate_positive_int(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_non_negative_int(value: int, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _validate_total_count(counts: Sequence[int], n_samples: int) -> None:
    total = sum(counts)
    if total > n_samples:
        raise ValueError(f"Requested {total} samples but dataset only has {n_samples}")


def _even_counts(n_samples: int, n_partitions: int) -> list[int]:
    quotient, remainder = divmod(n_samples, n_partitions)
    return [quotient + int(partition_index < remainder) for partition_index in range(n_partitions)]


def _split_by_counts(indices: Sequence[int], counts: Sequence[int]) -> list[list[int]]:
    partitions: list[list[int]] = []
    start = 0
    for count in counts:
        end = start + count
        partitions.append(list(indices[start:end]))
        start = end
    return partitions


def _normalize_labels(n_samples: int, labels: Sequence[Any] | None) -> list[Hashable]:
    _validate_non_negative_int(n_samples, "n_samples")
    if labels is None:
        raise ValueError("labels must be provided for this partitioner")
    if len(labels) != n_samples:
        raise ValueError(f"Expected {n_samples} labels, got {len(labels)}")
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
