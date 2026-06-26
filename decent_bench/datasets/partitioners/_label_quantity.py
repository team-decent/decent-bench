from collections import defaultdict, deque
from collections.abc import Hashable, Sequence

import numpy as np

from decent_bench.datasets._dataset_handler import DatasetHandler

from ._utils import (
    _even_counts,
    _indices_by_label,
    _normalized_labels,
    _partition_counts,
    _shuffled_sequence,
    _validate_positive_int,
)


def split_label_quantity(
    dataset: DatasetHandler,
    n_partitions: int,
    classes_per_partition: int,
    *,
    samples_per_partition: int | Sequence[int] | None = None,
) -> list[list[int]]:
    """
    Restrict each partition to a fixed number of labels.

    Args:
        dataset: Dataset handler whose indices are partitioned.
        n_partitions: Number of partitions to create.
        classes_per_partition: Maximum number of labels assigned to each partition.
        samples_per_partition: Optional common partition size or one size per partition.

    """
    _validate_positive_int(n_partitions, "n_partitions")
    _validate_positive_int(classes_per_partition, "classes_per_partition")

    label_to_indices = _indices_by_label(_normalized_labels(dataset))
    sorted_labels = sorted(label_to_indices, key=repr)
    if classes_per_partition > len(sorted_labels):
        raise ValueError("classes_per_partition must be <= the number of unique labels")

    label_order = _shuffled_sequence(sorted_labels)
    client_label_groups = [
        {
            label_order[(partition_index * classes_per_partition + offset) % len(label_order)]
            for offset in range(classes_per_partition)
        }
        for partition_index in range(n_partitions)
    ]

    clients_by_label: dict[Hashable, list[int]] = defaultdict(list)
    for partition_index, label_group in enumerate(client_label_groups):
        for label in label_group:
            clients_by_label[label].append(partition_index)

    usable_datapoints = sum(len(indices) for label, indices in label_to_indices.items() if clients_by_label[label])
    partition_counts = (
        _even_counts(usable_datapoints, n_partitions)
        if samples_per_partition is None
        else _partition_counts(usable_datapoints, n_partitions, samples_per_partition)
    )
    return _fill_allowed_label_partitions(label_to_indices, client_label_groups, partition_counts)


def _fill_allowed_label_partitions(
    label_to_indices: dict[Hashable, list[int]],
    allowed_labels_by_partition: Sequence[set[Hashable]],
    partition_counts: Sequence[int],
) -> list[list[int]]:
    partitions: list[list[int]] = [[] for _ in partition_counts]
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
    adjacency: list[list[int]] = [[] for _ in range(sink + 1)]
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
