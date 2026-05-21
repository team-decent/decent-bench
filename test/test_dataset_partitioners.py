import numpy as np
import pytest

import decent_bench.utils.interoperability as iop
from decent_bench.datasets import (
    DirichletLabelPartitioner,
    IidPartitioner,
    LabelQuantityPartitioner,
    PathologicalLabelPartitioner,
    PyTorchDatasetHandler,
    ShardPartitioner,
    SizePartitioner,
    SyntheticClassificationDatasetHandler,
)


def _flatten(partitions: list[list[int]]) -> list[int]:
    return [index for partition in partitions for index in partition]


def test_iid_partitioner_uses_all_indices_once() -> None:
    iop.set_seed(1)
    partitions = IidPartitioner(n_partitions=3).partition(10)

    assert [len(partition) for partition in partitions] == [4, 3, 3]
    assert sorted(_flatten(partitions)) == list(range(10))


def test_size_partitioner_uses_configured_partition_sizes() -> None:
    iop.set_seed(1)
    partitions = SizePartitioner([2, 0, 4]).partition(10)

    assert [len(partition) for partition in partitions] == [2, 0, 4]
    assert len(set(_flatten(partitions))) == 6
    assert all(0 <= index < 10 for index in _flatten(partitions))


def test_dirichlet_label_partitioner_preserves_all_indices() -> None:
    iop.set_seed(2)
    labels = [0] * 8 + [1] * 8 + [2] * 8
    partitions = DirichletLabelPartitioner(n_partitions=4, alpha=0.5).partition(len(labels), labels)

    assert sorted(_flatten(partitions)) == list(range(len(labels)))


def test_pathological_partitioner_assigns_label_sorted_shards() -> None:
    iop.set_seed(3)
    labels = [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4
    partitions = PathologicalLabelPartitioner(n_partitions=4, shards_per_partition=1).partition(len(labels), labels)

    assert [len(partition) for partition in partitions] == [4, 4, 4, 4]
    assert all(len({labels[index] for index in partition}) == 1 for partition in partitions)


def test_shard_partitioner_alias() -> None:
    assert issubclass(ShardPartitioner, PathologicalLabelPartitioner)


def test_label_quantity_partitioner_restricts_each_client_to_k_classes() -> None:
    iop.set_seed(4)
    labels = [0] * 6 + [1] * 6 + [2] * 6
    partitions = LabelQuantityPartitioner(n_partitions=6, classes_per_partition=1).partition(len(labels), labels)

    assert sorted(_flatten(partitions)) == list(range(len(labels)))
    assert all(len({labels[index] for index in partition}) <= 1 for partition in partitions)


def test_synthetic_classification_handler_accepts_partitioner() -> None:
    iop.set_seed(5)
    handler = SyntheticClassificationDatasetHandler(
        n_targets=2,
        n_features=5,
        n_samples_per_partition=8,
        partitioner=LabelQuantityPartitioner(n_partitions=3, classes_per_partition=1),
        squeeze_targets=True,
    )

    partitions = handler.get_partitions()

    assert handler.n_partitions == 3
    assert handler.n_samples == sum(len(partition) for partition in partitions)
    for partition in partitions:
        labels = {int(np.asarray(target)) for _, target in partition}
        assert len(labels) <= 1


def test_handler_rejects_partitioner_with_handler_n_partitions() -> None:
    with pytest.raises(ValueError, match="partitioner cannot be combined"):
        SyntheticClassificationDatasetHandler(
            n_targets=2,
            n_features=5,
            n_samples_per_partition=8,
            n_partitions=3,
            partitioner=IidPartitioner(n_partitions=3),
        )


def test_pytorch_handler_uses_partitioner_without_materializing_dataset() -> None:
    torch = pytest.importorskip("torch")

    class TinyDataset(torch.utils.data.Dataset):
        targets = [0, 0, 0, 1, 1, 1]

        def __len__(self) -> int:
            return len(self.targets)

        def __getitem__(self, index: int) -> tuple[object, object]:
            return np.array([index], dtype=np.float64), self.targets[index]

    handler = PyTorchDatasetHandler(
        TinyDataset(),
        n_features=1,
        n_targets=2,
        partitioner=LabelQuantityPartitioner(n_partitions=2, classes_per_partition=1),
    )

    partitions = handler.get_partitions()

    assert handler.n_partitions == 2
    assert [len(partition) for partition in partitions] == [3, 3]
    for partition in partitions:
        labels = {label for _, label in partition}
        assert len(labels) == 1
