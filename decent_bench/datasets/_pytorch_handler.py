from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

from decent_bench.utils.types import Dataset

from ._dataset_handler import DatasetHandler
from ._partitioners import IidPartitioner, Partitioner

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
        *,
        partitioner: Partitioner | None = None,
    ) -> None:
        """
        Dataset wrapper for PyTorch datasets which represents datapoints as tuples (features, targets).

        This class will preserve the properties of the underlying PyTorch dataset
        such as transforms and lazy-loading. Partitioning is controlled by the
        provided partitioner.

        Args:
            torch_dataset: PyTorch dataset to wrap
            n_features: Number of feature dimensions
            n_targets: Number of target dimensions
            partitioner: Optional partitioner defining the index split. If omitted, a single IID partition
                containing all samples is created.

        Raises:
            ImportError: If PyTorch is not installed

        Note:
            Use :class:`~decent_bench.datasets.IidPartitioner` for IID partitions and
            :class:`~decent_bench.datasets.LabelQuantityPartitioner` to restrict each partition
            to a fixed number of labels.

        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use PyTorchWrapper. Install it with: pip install torch")

        self.torch_dataset = torch_dataset
        self._n_targets = n_targets
        self._n_features = n_features
        self.partitioner = IidPartitioner(n_partitions=1) if partitioner is None else partitioner
        self._partitions: list[Dataset] | None = None

    @cached_property
    def n_samples(self) -> int:
        return sum(len(partition) for partition in self.get_partitions())

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

        Returns:
            Sequence[Dataset]: Sequence of Dataset objects, where each partition is a list of
            (features, targets) tuples.

        Note:
            The exact split is controlled by ``self.partitioner``. By default this is a single
            IID partition containing all samples.

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
