from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from decent_bench.utils.types import Dataset

from ._dataset_handler import DatasetHandler

if TYPE_CHECKING:
    import torch

try:
    import torch
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
    ) -> None:
        """
        Dataset wrapper for PyTorch datasets which represents datapoints as tuples (features, targets).

        This class preserves the properties of the underlying PyTorch dataset,
        such as transforms and lazy loading.

        Args:
            torch_dataset: PyTorch dataset to wrap
            n_features: Number of feature dimensions
            n_targets: Number of target dimensions

        Raises:
            ImportError: If PyTorch is not installed

        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use PyTorchWrapper. Install it with: pip install torch")

        self.torch_dataset = torch_dataset
        self._n_targets = n_targets
        self._n_features = n_features

    @property
    def n_samples(self) -> int:
        return len(self.torch_dataset)  # type: ignore[arg-type]

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
        return cast("Dataset", [self.torch_dataset[index] for index in range(self.n_samples)])

    def get_labels(self) -> list[Any]:
        """Return labels from the underlying dataset."""
        dataset_labels = getattr(self.torch_dataset, "targets", None)
        if dataset_labels is None:
            dataset_labels = getattr(self.torch_dataset, "labels", None)
        if dataset_labels is not None:
            if hasattr(dataset_labels, "tolist"):
                labels = dataset_labels.tolist()
                return labels if isinstance(labels, list) else [labels]
            return list(cast("Sequence[Any]", dataset_labels))

        return [self.torch_dataset[index][1] for index in range(self.n_samples)]

    def split(
        self,
        partitions: Sequence[Sequence[int]],
    ) -> list[Dataset]:
        """Materialize index partitions as lazy PyTorch subsets."""
        idx_partitions = self._resolve_partitions(partitions)
        materialized_partitions = [TorchSubset(self.torch_dataset, indices) for indices in idx_partitions]
        return cast("list[Dataset]", materialized_partitions)
