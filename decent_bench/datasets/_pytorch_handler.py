from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

from decent_bench.utils.types import Dataset

from ._dataset_handler import DatasetHandler
from ._partitioners import IidPartitioner, LabelQuantityPartitioner, Partitioner

if TYPE_CHECKING:
    import torch

try:
    import torch
    from torch.utils.data import ConcatDataset as TorchConcatDataset
    from torch.utils.data import Subset as TorchSubset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class PyTorchDatasetHandler(DatasetHandler):
    def __init__(
        self,
        torch_dataset: torch.utils.data.Dataset[Any],
        n_features: int,
        n_targets: int,
        n_partitions: int | None = None,
        *,
        samples_per_partition: int | None = None,
        heterogeneity: bool = False,
        targets_per_partition: int = 1,
        partitioner: Partitioner | None = None,
    ) -> None:
        """
        Dataset wrapper for PyTorch datasets which represents datapoints as tuples (features, targets).

        This class will preserve the properties of the underlying PyTorch dataset
        such as transforms and lazy-loading. This class can create either random partitions where
        each partition is drawn uniformly at random from the dataset without replacement (heterogeneity=False),
        or heterogeneous partitions (heterogeneity=True) where each partition contains unique classes.
        Heterogeneity only works for datasets where the targets are categorical.

        Args:
            torch_dataset: PyTorch dataset to wrap
            n_features: Number of feature dimensions
            n_targets: Number of target dimensions
            n_partitions: Number of partitions for the default partitioner. Do not use together with partitioner.
            samples_per_partition: Number of samples per partition for the default IID partitioner.
            heterogeneity: Whether to use the default label-quantity partitioner with unique classes.
            targets_per_partition: Number of unique classes per partition for the default heterogeneous partitioner.
            partitioner: Optional partitioner defining the index split. If omitted, an appropriate default
                partitioner is created from the legacy partitioning arguments.

        Raises:
            ImportError: If PyTorch is not installed
            ValueError: If heterogeneity is True and n_partitions * targets_per_partition > n_targets

        Note:
            If heterogeneity is True, each partition will contain unique classes.
            Ensure that n_partitions * targets_per_partition <= n_targets. Be aware that
            this may lead to some classes being unused if the condition is not tight,
            the :meth:`n_targets` attribute will be updated accordingly.

            If the underlying PyTorch dataset has not implemented __len__, set samples_per_partition
            to specify the number of samples per partition or set heterogeneity to True. Otherwise,
            the length of the dataset cannot be determined and an error will be raised.

        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use PyTorchWrapper. Install it with: pip install torch")

        if partitioner is not None:
            if n_partitions is not None or samples_per_partition is not None or heterogeneity:
                raise ValueError(
                    "partitioner cannot be combined with n_partitions, samples_per_partition, or heterogeneity"
                )
            if targets_per_partition != 1:
                raise ValueError("targets_per_partition must be configured on the partitioner when partitioner is set")
        else:
            n_partitions = 1 if n_partitions is None else n_partitions
            if heterogeneity:
                if (n_partitions * targets_per_partition) > n_targets:
                    raise ValueError(
                        f"n_partitions ({n_partitions}) * n_targets per partition ({targets_per_partition})"
                        f" must be <= n_targets ({n_targets})"
                    )
                partitioner = LabelQuantityPartitioner(
                    n_partitions=n_partitions,
                    classes_per_partition=targets_per_partition,
                    samples_per_partition=samples_per_partition,
                )
            else:
                partitioner = IidPartitioner(
                    n_partitions=n_partitions,
                    samples_per_partition=samples_per_partition,
                )

        self.torch_dataset = torch_dataset
        self._n_targets = n_targets
        self._n_features = n_features
        self.partitioner = partitioner
        self._partitions: list[Dataset] | None = None

        if heterogeneity:
            # Set the new number of used targets
            self._n_targets = self.n_partitions * targets_per_partition

    @cached_property
    def n_samples(self) -> int:
        return len(self.get_datapoints())

    @property
    def n_partitions(self) -> int:
        return self.partitioner.n_partitions

    @property
    def n_features(self) -> int:
        return self._n_features

    @property
    def n_targets(self) -> int:
        return self._n_targets

    def get_datapoints(self) -> Dataset:
        """
        Return all datapoints in the dataset.

        Can be used for evaluation on the full dataset or creation of test datasets.
        """
        return cast("Dataset", list(TorchConcatDataset(self.get_partitions())))  # type: ignore[arg-type, call-overload]

    def get_partitions(self) -> list[Dataset]:
        """
        Return the dataset divided into partitions for distribution among agents.

        This method provides the core partitioning functionality for decentralized
        optimization. Each partition represents the local dataset of an agent in
        the network.

        Each partition is sampled uniformly at random from the dataset without replacement
        if heterogeneity is False, otherwise each partition contains unique classes (targets_per_partition)
        with number of datapoints per partition equal to
        min(samples_per_partition, number of available datapoints for the selected classes).

        Returns:
            Sequence[Dataset]: Sequence of Dataset objects, where each partition is a list of
            (features, targets) tuples.

        """
        if self._partitions is None:
            self._partitions = self._partitioner_split()

        return self._partitions

    def _partitioner_split(self) -> list[Dataset]:
        labels = self._get_labels() if self.partitioner.requires_labels else None
        idx_partitions = self.partitioner.partition(len(self.torch_dataset), labels=labels)  # type: ignore[arg-type]
        partitions = [TorchSubset(self.torch_dataset, indices) for indices in idx_partitions]
        return cast("list[Dataset]", partitions)

    def _get_labels(self) -> list[Any]:
        dataset_labels = getattr(self.torch_dataset, "targets", None)
        if dataset_labels is None:
            dataset_labels = getattr(self.torch_dataset, "labels", None)
        if dataset_labels is not None:
            if hasattr(dataset_labels, "tolist"):
                labels = dataset_labels.tolist()
                return labels if isinstance(labels, list) else [labels]
            return list(cast("Sequence[Any]", dataset_labels))

        return [label for _, label in cast("Sequence[tuple[Any, Any]]", self.torch_dataset)]
