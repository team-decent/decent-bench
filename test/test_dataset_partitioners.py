from collections import Counter
from collections.abc import Sequence

import numpy as np
import pytest

import decent_bench.utils.interoperability as iop
from decent_bench.datasets import (
    DatasetHandler,
    PyTorchDatasetHandler,
    SyntheticClassificationDatasetHandler,
    split_dirichlet_label,
    split_iid,
    split_label_quantity,
    split_shard,
    split_size,
    split_stratified_iid,
)
from decent_bench.utils.types import Dataset


class ListDatasetHandler(DatasetHandler):
    def __init__(self, labels: Sequence[int]) -> None:
        self._dataset: Dataset = [(np.asarray([index]), np.asarray(label)) for index, label in enumerate(labels)]

    @property
    def n_samples(self) -> int:
        return len(self._dataset)

    @property
    def n_features(self) -> int:
        return 1

    @property
    def n_targets(self) -> int:
        return 1

    def get_datapoints(self) -> Dataset:
        return self._dataset

    def get_labels(self) -> list[int]:
        return [int(np.asarray(target)) for _, target in self._dataset]

    def split(
        self,
        partitions: Sequence[Sequence[int]],
    ) -> list[Dataset]:
        idx_partitions = self._resolve_partitions(partitions)
        return [[self._dataset[index] for index in partition] for partition in idx_partitions]


def _dataset(labels: Sequence[int]) -> ListDatasetHandler:
    return ListDatasetHandler(labels)


def _flatten(partitions: list[list[int]]) -> list[int]:
    return [index for partition in partitions for index in partition]


def _label_counts(partition: list[int], labels: list[int], label_values: list[int]) -> list[int]:
    counts = Counter(labels[index] for index in partition)
    return [counts[label] for label in label_values]


def _label_proportions(partition: list[int], labels: list[int], label_values: list[int]) -> list[float]:
    counts = _label_counts(partition, labels, label_values)
    return [count / len(partition) for count in counts]


def _global_label_proportions(labels: list[int], label_values: list[int]) -> list[float]:
    counts = Counter(labels)
    return [counts[label] / len(labels) for label in label_values]


def _total_variation_distance(p: list[float], q: list[float]) -> float:
    return 0.5 * sum(abs(p_i - q_i) for p_i, q_i in zip(p, q, strict=True))


def _assert_has_label_skew(
    partitions: list[list[int]],
    labels: list[int],
    label_values: list[int],
    min_total_variation_distance: float,
) -> None:
    global_proportions = _global_label_proportions(labels, label_values)
    partition_distances = [
        _total_variation_distance(_label_proportions(partition, labels, label_values), global_proportions)
        for partition in partitions
        if len(partition) > 0
    ]
    if not partition_distances or max(partition_distances) <= min_total_variation_distance:
        pytest.fail("Expected at least one partition to have label skew above the TVD threshold")


def test_split_iid_preserves_global_label_distribution_approximately() -> None:
    iop.set_seed(1)
    labels = [0] * 300 + [1] * 300 + [2] * 300
    label_values = [0, 1, 2]
    partitions = split_iid(_dataset(labels), n_partitions=3)

    assert [len(partition) for partition in partitions] == [300, 300, 300]
    assert sorted(_flatten(partitions)) == list(range(len(labels)))
    for partition in partitions:
        proportions = _label_proportions(partition, labels, label_values)
        assert all(abs(proportion - (1 / 3)) < 0.08 for proportion in proportions)


def test_split_iid_can_use_explicit_partition_sizes() -> None:
    iop.set_seed(2)
    partitions = split_iid(_dataset([0] * 10), n_partitions=3, samples_per_partition=[2, 0, 4])

    assert [len(partition) for partition in partitions] == [2, 0, 4]
    assert len(set(_flatten(partitions))) == 6
    assert all(0 <= index < 10 for index in _flatten(partitions))


def test_split_stratified_iid_balances_each_label_across_partitions() -> None:
    iop.set_seed(2)
    labels = [0] * 10 + [1] * 10 + [2] * 10
    label_values = [0, 1, 2]
    partitions = split_stratified_iid(_dataset(labels), n_partitions=5)

    assert [len(partition) for partition in partitions] == [6, 6, 6, 6, 6]
    assert sorted(_flatten(partitions)) == list(range(len(labels)))
    for partition in partitions:
        assert _label_counts(partition, labels, label_values) == [2, 2, 2]


def test_split_stratified_iid_can_limit_samples_per_partition() -> None:
    iop.set_seed(3)
    labels = [0] * 10 + [1] * 10 + [2] * 10
    label_values = [0, 1, 2]
    partitions = split_stratified_iid(_dataset(labels), n_partitions=5, samples_per_partition=3)

    assert [len(partition) for partition in partitions] == [3, 3, 3, 3, 3]
    assert len(set(_flatten(partitions))) == 15
    for partition in partitions:
        assert _label_counts(partition, labels, label_values) == [1, 1, 1]


def test_split_stratified_iid_can_use_explicit_partition_sizes() -> None:
    iop.set_seed(4)
    labels = [0] * 12 + [1] * 12 + [2] * 12
    label_values = [0, 1, 2]
    partitions = split_stratified_iid(
        _dataset(labels),
        n_partitions=3,
        samples_per_partition=[3, 6, 9],
    )

    assert [len(partition) for partition in partitions] == [3, 6, 9]
    assert len(set(_flatten(partitions))) == 18
    assert _label_counts(partitions[0], labels, label_values) == [1, 1, 1]
    assert _label_counts(partitions[1], labels, label_values) == [2, 2, 2]
    assert _label_counts(partitions[2], labels, label_values) == [3, 3, 3]


def test_split_size_uses_configured_partition_sizes() -> None:
    iop.set_seed(1)
    partitions = split_size(_dataset([0] * 10), partition_sizes=[2, 0, 4])

    assert [len(partition) for partition in partitions] == [2, 0, 4]
    assert len(set(_flatten(partitions))) == 6
    assert all(0 <= index < 10 for index in _flatten(partitions))


def test_split_dirichlet_label_creates_label_skew() -> None:
    iop.set_seed(2)
    labels = [0] * 300 + [1] * 300 + [2] * 300
    label_values = [0, 1, 2]
    partitions = split_dirichlet_label(
        _dataset(labels),
        n_partitions=6,
        alpha=0.05,
        min_partition_size=1,
        max_retries=1000,
    )

    assert sorted(_flatten(partitions)) == list(range(len(labels)))
    _assert_has_label_skew(partitions, labels, label_values, min_total_variation_distance=0.6)
    assert any(max(_label_proportions(partition, labels, label_values)) > 0.95 for partition in partitions)


def test_split_dirichlet_label_can_use_explicit_partition_sizes() -> None:
    iop.set_seed(6)
    labels = [0] * 300 + [1] * 300 + [2] * 300
    partitions = split_dirichlet_label(
        _dataset(labels),
        n_partitions=4,
        alpha=0.05,
        samples_per_partition=[40, 80, 120, 160],
        min_partition_size=1,
        max_retries=1000,
    )

    assert [len(partition) for partition in partitions] == [40, 80, 120, 160]
    assert len(set(_flatten(partitions))) == 400
    assert any(len({labels[index] for index in partition}) == 1 for partition in partitions)


def test_split_shard_assigns_label_sorted_shards() -> None:
    iop.set_seed(3)
    labels = [0] * 40 + [1] * 40 + [2] * 40 + [3] * 40
    label_values = [0, 1, 2, 3]
    partitions = split_shard(_dataset(labels), n_partitions=4, shards_per_partition=1)

    assert [len(partition) for partition in partitions] == [40, 40, 40, 40]
    assert sorted(_flatten(partitions)) == list(range(len(labels)))
    _assert_has_label_skew(partitions, labels, label_values, min_total_variation_distance=0.7)
    assert all(len({labels[index] for index in partition}) == 1 for partition in partitions)


def test_split_label_quantity_restricts_each_client_to_k_classes() -> None:
    iop.set_seed(4)
    labels = [0] * 60 + [1] * 60 + [2] * 60
    label_values = [0, 1, 2]
    partitions = split_label_quantity(_dataset(labels), n_partitions=6, classes_per_partition=1)

    assert sorted(_flatten(partitions)) == list(range(len(labels)))
    _assert_has_label_skew(partitions, labels, label_values, min_total_variation_distance=0.6)
    assert all(len({labels[index] for index in partition}) <= 1 for partition in partitions)
    assert all(len(partition) == 30 for partition in partitions)


def test_split_label_quantity_can_use_explicit_partition_sizes() -> None:
    iop.set_seed(7)
    labels = [0] * 80 + [1] * 80 + [2] * 80
    partitions = split_label_quantity(
        _dataset(labels),
        n_partitions=4,
        classes_per_partition=3,
        samples_per_partition=[10, 20, 30, 40],
    )

    assert [len(partition) for partition in partitions] == [10, 20, 30, 40]
    assert len(set(_flatten(partitions))) == 100
    assert all(len({labels[index] for index in partition}) <= 3 for partition in partitions)


def test_split_label_quantity_can_limit_every_partition_to_same_size() -> None:
    iop.set_seed(8)
    labels = [0] * 60 + [1] * 60 + [2] * 60
    partitions = split_label_quantity(
        _dataset(labels),
        n_partitions=6,
        classes_per_partition=1,
        samples_per_partition=20,
    )

    assert [len(partition) for partition in partitions] == [20, 20, 20, 20, 20, 20]
    assert all(len({labels[index] for index in partition}) <= 1 for partition in partitions)


def test_split_label_quantity_finds_feasible_non_greedy_assignment() -> None:
    iop.set_seed(9)
    labels = [0, 1, 2]
    partitions = split_label_quantity(_dataset(labels), n_partitions=3, classes_per_partition=2)

    assert [len(partition) for partition in partitions] == [1, 1, 1]
    assert sorted(_flatten(partitions)) == list(range(len(labels)))


def test_split_label_quantity_raises_when_default_even_split_is_infeasible() -> None:
    iop.set_seed(8)
    labels = [0] * 60 + [1] * 60 + [2]

    with pytest.raises(ValueError, match="Could not create label-quantity partitions"):
        split_label_quantity(_dataset(labels), n_partitions=3, classes_per_partition=1)


def test_dataset_split_materializes_explicit_indices() -> None:
    dataset = _dataset([0, 1, 0, 1])

    partitions = dataset.split(partitions=[[3, 1], [0]])

    assert [[int(features[0]) for features, _ in partition] for partition in partitions] == [[3, 1], [0]]
    assert dataset.n_samples == 4
    assert len(dataset.get_datapoints()) == 4


def test_dataset_split_requires_explicit_partitions() -> None:
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        _dataset([0, 1]).split()  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("partitions", "error_type", "match"),
    [
        ([[0], [0]], ValueError, "occurs more than once"),
        ([[4]], IndexError, "out of bounds"),
        ([[True]], TypeError, "must be integers"),
    ],
)
def test_dataset_split_validates_explicit_indices(
    partitions: list[list[int]],
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        _dataset([0, 1, 0, 1]).split(partitions=partitions)


def test_synthetic_classification_handler_works_with_label_quantity_split() -> None:
    iop.set_seed(5)
    dataset = SyntheticClassificationDatasetHandler(
        n_targets=2,
        n_features=5,
        n_samples=24,
        squeeze_targets=True,
    )
    idx_partitions = split_label_quantity(
        dataset,
        n_partitions=3,
        classes_per_partition=1,
        samples_per_partition=4,
    )

    partitions = dataset.split(partitions=idx_partitions)

    assert [len(partition) for partition in partitions] == [4, 4, 4]
    assert dataset.n_samples == 24
    for partition in partitions:
        labels = {int(np.asarray(target)) for _, target in partition}
        assert len(labels) <= 1


def test_pytorch_handler_materializes_partitions_as_subsets() -> None:
    torch = pytest.importorskip("torch")

    class TinyDataset(torch.utils.data.Dataset):
        targets = [0, 0, 0, 1, 1, 1]

        def __len__(self) -> int:
            return len(self.targets)

        def __getitem__(self, index: int) -> tuple[object, object]:
            return np.array([index], dtype=np.float64), self.targets[index]

    dataset = PyTorchDatasetHandler(TinyDataset(), n_features=1, n_targets=2)
    idx_partitions = split_label_quantity(dataset, n_partitions=2, classes_per_partition=1)

    partitions = dataset.split(partitions=idx_partitions)

    assert [len(partition) for partition in partitions] == [3, 3]
    for partition in partitions:
        labels = {label for _, label in partition}
        assert len(labels) == 1
