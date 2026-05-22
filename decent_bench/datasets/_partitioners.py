from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

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
    def partition(self, n_datapoints: int, labels: Sequence[Any] | None = None) -> list[list[int]]:
        """Return index partitions for a dataset with ``n_datapoints`` datapoints."""


class IidPartitioner(Partitioner):
    """
    Randomly split datapoints into IID partitions.

    This partitioner shuffles datapoint indices and splits them without replacement,
    so partitions are IID in expectation with respect to the global empirical
    distribution. It does not inspect labels and does not guarantee class balance
    in each partition. Use :class:`StratifiedIidPartitioner` when every partition
    should preserve label proportions explicitly.
    """

    def __init__(self, n_partitions: int, samples_per_partition: int | Sequence[int] | None = None) -> None:
        """
        Initialize an IID partitioner.

        Args:
            n_partitions: Number of partitions to create.
            samples_per_partition: Optional fixed number of samples in every partition,
                or one explicit size per partition.

        """
        super().__init__(n_partitions)
        self.samples_per_partition = _validate_samples_per_partition(samples_per_partition, n_partitions)

    def partition(
        self,
        n_datapoints: int,
        labels: Sequence[Any] | None = None,  # noqa: ARG002
    ) -> list[list[int]]:
        """Return random partitions sampled without replacement."""
        counts = _partition_counts(n_datapoints, self.n_partitions, self.samples_per_partition)

        indices = _shuffled_range(n_datapoints)
        return _split_by_counts(indices, counts)


class StratifiedIidPartitioner(Partitioner):
    """
    Split every label as evenly as possible across IID partitions.

    This is the label-balanced IID partitioner: it requires labels, groups
    datapoints by label, and distributes each label proportionally across
    partitions before shuffling within each partition.
    """

    def __init__(self, n_partitions: int, samples_per_partition: int | Sequence[int] | None = None) -> None:
        """
        Initialize a stratified IID partitioner.

        Args:
            n_partitions: Number of partitions to create.
            samples_per_partition: Optional fixed number of samples in every partition,
                or one explicit size per partition.

        """
        super().__init__(n_partitions)
        self.samples_per_partition = _validate_samples_per_partition(samples_per_partition, n_partitions)

    @property
    def requires_labels(self) -> bool:
        """Whether this partitioner requires one label per datapoint."""
        return True

    def partition(self, n_datapoints: int, labels: Sequence[Any] | None = None) -> list[list[int]]:
        """
        Return partitions with label counts balanced across clients.

        Raises:
            ValueError: If labels are missing or the requested partition sizes exceed the dataset size.

        """
        normalized_labels = _normalize_labels(n_datapoints, labels)
        partition_counts = _partition_counts(n_datapoints, self.n_partitions, self.samples_per_partition)

        label_items = sorted(_indices_by_label(normalized_labels).items(), key=lambda item: repr(item[0]))
        label_counts = _proportional_counts(
            [len(indices) for _, indices in label_items],
            sum(partition_counts),
        )

        partitions: list[list[int]] = [[] for _ in range(self.n_partitions)]
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

    def partition(
        self,
        n_datapoints: int,
        labels: Sequence[Any] | None = None,  # noqa: ARG002
    ) -> list[list[int]]:
        """Return random partitions with the configured sizes."""
        counts = _partition_counts(n_datapoints, self.n_partitions, self.partition_sizes)

        indices = _shuffled_range(n_datapoints)
        return _split_by_counts(indices, counts)


class DirichletLabelPartitioner(Partitioner):
    """Split labels across partitions using a Dirichlet distribution."""

    def __init__(
        self,
        n_partitions: int,
        alpha: float,
        *,
        samples_per_partition: int | Sequence[int] | None = None,
        min_partition_size: int = 0,
        max_retries: int = 100,
    ) -> None:
        """
        Initialize a Dirichlet label-skew partitioner.

        Args:
            n_partitions: Number of partitions to create.
            alpha: Dirichlet concentration. Smaller values create stronger label skew.
            samples_per_partition: Optional fixed number of samples in every partition,
                or one explicit size per partition.
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
        self.samples_per_partition = _validate_samples_per_partition(samples_per_partition, n_partitions)
        self.min_partition_size = min_partition_size
        self.max_retries = max_retries

    @property
    def requires_labels(self) -> bool:
        """Whether this partitioner requires one label per datapoint."""
        return True

    def partition(self, n_datapoints: int, labels: Sequence[Any] | None = None) -> list[list[int]]:
        """
        Return label-skewed partitions sampled without replacement.

        Raises:
            ValueError: If labels are missing or the minimum partition size cannot be satisfied.

        """
        normalized_labels = _normalize_labels(n_datapoints, labels)
        partition_counts = _partition_counts(n_datapoints, self.n_partitions, self.samples_per_partition)
        for _ in range(self.max_retries):
            partitions = self._partition_once(normalized_labels, partition_counts)
            if min(len(partition) for partition in partitions) >= self.min_partition_size:
                return partitions

        raise ValueError(
            f"Could not create {self.n_partitions} partitions with min_partition_size "
            f"{self.min_partition_size} after {self.max_retries} attempts"
        )

    def _partition_once(self, labels: Sequence[Hashable], partition_counts: Sequence[int]) -> list[list[int]]:
        partitions: list[list[int]] = [[] for _ in range(self.n_partitions)]
        remaining_counts = list(partition_counts)
        label_items = list(_indices_by_label(labels).items())
        label_counts = _proportional_counts(
            [len(indices) for _, indices in label_items],
            sum(partition_counts),
        )
        rng = iop.rng_numpy()

        label_assignments = _shuffled_sequence(list(zip(label_items, label_counts, strict=True)))
        for (_, label_indices), label_count in label_assignments:
            selected_indices = _shuffled_sequence(label_indices)[:label_count]
            proportions = rng.dirichlet(np.full(self.n_partitions, self.alpha))
            _extend_weighted_by_capacity(selected_indices, partitions, remaining_counts, proportions)

        if any(remaining_counts):
            raise ValueError("Could not create Dirichlet partitions with the requested sizes")

        return [_shuffled_sequence(partition) for partition in partitions]


class ShardPartitioner(Partitioner):
    """Sort datapoints by label, cut them into shards, and assign shards to clients."""

    def __init__(
        self,
        n_partitions: int,
        shards_per_partition: int,
    ) -> None:
        """
        Initialize a shard-based pathological label partitioner.

        Args:
            n_partitions: Number of partitions to create.
            shards_per_partition: Number of label-sorted shards assigned to each partition.

        """
        super().__init__(n_partitions)
        _validate_positive_int(shards_per_partition, "shards_per_partition")
        self.shards_per_partition = shards_per_partition

    @property
    def requires_labels(self) -> bool:
        """Whether this partitioner requires one label per datapoint."""
        return True

    def partition(self, n_datapoints: int, labels: Sequence[Any] | None = None) -> list[list[int]]:
        """
        Return partitions made from randomly assigned label-sorted shards.

        Raises:
            ValueError: If labels are missing or more shards than datapoints are requested.

        """
        normalized_labels = _normalize_labels(n_datapoints, labels)
        total_shards = self.n_partitions * self.shards_per_partition
        if total_shards > n_datapoints:
            raise ValueError("n_partitions * shards_per_partition must be <= n_datapoints")

        sorted_indices = sorted(range(n_datapoints), key=lambda index: repr(normalized_labels[index]))
        shards = [[int(index) for index in shard] for shard in np.array_split(sorted_indices, total_shards)]
        shard_order = _shuffled_range(total_shards)

        partitions: list[list[int]] = []
        for partition_index in range(self.n_partitions):
            start = partition_index * self.shards_per_partition
            assigned_shards = shard_order[start : start + self.shards_per_partition]
            partition = [index for shard_index in assigned_shards for index in shards[shard_index]]
            partitions.append(_shuffled_sequence(partition))

        return partitions


class LabelQuantityPartitioner(Partitioner):
    """Restrict each partition to a fixed number of labels."""

    def __init__(
        self,
        n_partitions: int,
        classes_per_partition: int,
        *,
        samples_per_partition: int | Sequence[int] | None = None,
    ) -> None:
        """
        Initialize a label-quantity partitioner.

        Args:
            n_partitions: Number of partitions to create.
            classes_per_partition: Number of classes allowed in each partition.
            samples_per_partition: Optional fixed number of samples in every partition,
                or one explicit size per partition. If omitted, an even split
                over usable datapoints is attempted and a ``ValueError`` is
                raised when the label constraints make that split infeasible.

        """
        super().__init__(n_partitions)
        _validate_positive_int(classes_per_partition, "classes_per_partition")

        self.classes_per_partition = classes_per_partition
        self.samples_per_partition = _validate_samples_per_partition(samples_per_partition, n_partitions)

    @property
    def requires_labels(self) -> bool:
        """Whether this partitioner requires one label per datapoint."""
        return True

    def partition(self, n_datapoints: int, labels: Sequence[Any] | None = None) -> list[list[int]]:
        """
        Return partitions where each client sees at most ``classes_per_partition`` labels.

        Raises:
            ValueError: If labels are missing, too many classes per partition are requested,
                or the requested/default partition sizes cannot be filled under the label constraints.

        """
        normalized_labels = _normalize_labels(n_datapoints, labels)
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

        usable_datapoints = sum(len(indices) for label, indices in label_to_indices.items() if clients_by_label[label])
        if self.samples_per_partition is None:
            partition_counts = _even_counts(usable_datapoints, self.n_partitions)
            return _fill_allowed_label_partitions(label_to_indices, client_label_groups, partition_counts)

        partition_counts = _partition_counts(usable_datapoints, self.n_partitions, self.samples_per_partition)
        return _fill_allowed_label_partitions(label_to_indices, client_label_groups, partition_counts)


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


def _validate_total_count(counts: Sequence[int], n_datapoints: int) -> None:
    total = sum(counts)
    if total > n_datapoints:
        raise ValueError(f"Requested {total} datapoints but dataset only has {n_datapoints}")


def _partition_counts(
    n_datapoints: int,
    n_partitions: int,
    samples_per_partition: int | Sequence[int] | None,
) -> list[int]:
    _validate_non_negative_int(n_datapoints, "n_datapoints")
    normalized_samples_per_partition = _validate_samples_per_partition(samples_per_partition, n_partitions)
    if normalized_samples_per_partition is None:
        counts = _even_counts(n_datapoints, n_partitions)
    elif isinstance(normalized_samples_per_partition, int):
        counts = [normalized_samples_per_partition] * n_partitions
    else:
        counts = normalized_samples_per_partition

    _validate_total_count(counts, n_datapoints)
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
        if available_weight_sum == 0:
            probabilities = np.full(len(available_partitions), 1 / len(available_partitions))
        else:
            probabilities = available_weights / available_weight_sum

        selected_partition = int(rng.choice(np.asarray(available_partitions), p=probabilities))
        partitions[selected_partition].append(index)
        remaining_counts[selected_partition] -= 1


def _fill_allowed_label_partitions(
    label_to_indices: dict[Hashable, list[int]],
    allowed_labels_by_partition: Sequence[set[Hashable]],
    partition_counts: Sequence[int],
) -> list[list[int]]:
    partitions: list[list[int]] = [[] for _ in range(len(partition_counts))]
    remaining_indices_by_label = {
        label: _shuffled_sequence(indices)
        for label, indices in label_to_indices.items()
        if any(label in allowed_labels for allowed_labels in allowed_labels_by_partition)
    }
    allocations = _label_partition_allocations(
        remaining_indices_by_label,
        allowed_labels_by_partition,
        partition_counts,
    )

    for label, counts in allocations.items():
        start = 0
        indices = remaining_indices_by_label[label]
        for partition_index, count in enumerate(counts):
            end = start + count
            partitions[partition_index].extend(indices[start:end])
            start = end

    return [_shuffled_sequence(partition) for partition in partitions]


def _label_partition_allocations(
    indices_by_label: dict[Hashable, list[int]],
    allowed_labels_by_partition: Sequence[set[Hashable]],
    partition_counts: Sequence[int],
) -> dict[Hashable, list[int]]:
    labels = sorted(indices_by_label, key=repr)
    n_labels = len(labels)
    n_partitions = len(partition_counts)
    source = 0
    label_node_offset = 1
    partition_node_offset = label_node_offset + n_labels
    sink = partition_node_offset + n_partitions
    n_nodes = sink + 1

    adjacency: list[list[int]] = [[] for _ in range(n_nodes)]
    capacities: dict[tuple[int, int], int] = {}

    for label_index, label in enumerate(labels):
        _add_flow_edge(
            adjacency,
            capacities,
            source,
            label_node_offset + label_index,
            len(indices_by_label[label]),
        )

    total_requested = sum(partition_counts)
    for label_index, label in enumerate(labels):
        label_node = label_node_offset + label_index
        for partition_index, allowed_labels in enumerate(allowed_labels_by_partition):
            if label in allowed_labels:
                _add_flow_edge(
                    adjacency,
                    capacities,
                    label_node,
                    partition_node_offset + partition_index,
                    total_requested,
                )

    for partition_index, partition_count in enumerate(partition_counts):
        _add_flow_edge(
            adjacency,
            capacities,
            partition_node_offset + partition_index,
            sink,
            partition_count,
        )

    if _max_flow(adjacency, capacities, source, sink) != total_requested:
        raise ValueError("Could not create label-quantity partitions with the requested sizes")

    allocations: dict[Hashable, list[int]] = {}
    for label_index, label in enumerate(labels):
        label_node = label_node_offset + label_index
        allocations[label] = [
            capacities.get((partition_node_offset + partition_index, label_node), 0)
            for partition_index in range(n_partitions)
        ]

    return allocations


def _add_flow_edge(
    adjacency: list[list[int]],
    capacities: dict[tuple[int, int], int],
    source: int,
    target: int,
    capacity: int,
) -> None:
    if capacity == 0:
        return
    adjacency[source].append(target)
    adjacency[target].append(source)
    capacities[source, target] = capacities.get((source, target), 0) + capacity
    capacities.setdefault((target, source), 0)


def _max_flow(
    adjacency: list[list[int]],
    capacities: dict[tuple[int, int], int],
    source: int,
    sink: int,
) -> int:
    flow = 0
    while True:
        parent = _find_augmenting_path(adjacency, capacities, source, sink)
        if parent[sink] == -1:
            return flow

        path_flow = np.iinfo(np.int64).max
        node = sink
        while node != source:
            previous = parent[node]
            path_flow = min(path_flow, capacities[previous, node])
            node = previous

        node = sink
        while node != source:
            previous = parent[node]
            capacities[previous, node] -= path_flow
            capacities[node, previous] += path_flow
            node = previous

        flow += path_flow


def _find_augmenting_path(
    adjacency: list[list[int]],
    capacities: dict[tuple[int, int], int],
    source: int,
    sink: int,
) -> list[int]:
    parent = [-1] * len(adjacency)
    parent[source] = source
    queue = deque([source])

    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if parent[neighbor] != -1 or capacities[node, neighbor] <= 0:
                continue
            parent[neighbor] = node
            if neighbor == sink:
                return parent
            queue.append(neighbor)

    return parent


def _normalize_labels(n_datapoints: int, labels: Sequence[Any] | None) -> list[Hashable]:
    _validate_non_negative_int(n_datapoints, "n_datapoints")
    if labels is None:
        raise ValueError("labels must be provided for this partitioner")
    if len(labels) != n_datapoints:
        raise ValueError(f"Expected {n_datapoints} labels, got {len(labels)}")
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
