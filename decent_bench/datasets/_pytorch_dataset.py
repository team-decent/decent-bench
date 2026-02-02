from __future__ import annotations

import random
from collections import defaultdict
from typing import TYPE_CHECKING, Any, cast

from decent_bench.utils.logger import LOGGER
from decent_bench.utils.types import DatasetPartition

from ._dataset import Dataset

if TYPE_CHECKING:
    import torch

try:
    from torch import Generator
    from torch.utils.data import Subset as TorchSubset
    from torch.utils.data import random_split as torch_random_split

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class PyTorchDataset(Dataset):
    def __init__(
        self,
        torch_dataset: torch.utils.data.Dataset[Any],
        n_features: int,
        n_targets: int,
        n_partitions: int,
        *,
        samples_per_partition: int | None = None,
        heterogeneity: bool = False,
        targets_per_partition: int = 1,
        seed: int | None = None,
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
            n_partitions: Number of partitions to split the dataset into
            samples_per_partition: Number of samples per partition, if None, will split evenly
            heterogeneity: Whether to create heterogeneous partitions with unique classes
            targets_per_partition: Number of unique classes per partition (only if heterogeneity is True)
            seed: Random seed for shuffling the dataset

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
        self.torch_dataset = torch_dataset
        self._n_targets = n_targets
        self._n_features = n_features
        self._n_partitions = n_partitions
        self.samples_per_partition = samples_per_partition
        self.heterogeneity = heterogeneity
        self.targets_per_partition = targets_per_partition
        self.seed = seed
        self._partitions: list[DatasetPartition] | None = None

        if self.heterogeneity:
            if (self.n_partitions * self.targets_per_partition) > self.n_targets:
                raise ValueError(
                    f"n_partitions ({self.n_partitions}) * n_targets per partition ({self.targets_per_partition})"
                    f" must be <= n_targets ({self.n_targets})"
                )
            # Set the new number of used targets
            self._n_targets = self.n_partitions * self.targets_per_partition

    @property
    def n_samples(self) -> int:
        return len(self.torch_dataset)  # type: ignore[arg-type]

    @property
    def n_partitions(self) -> int:
        return self._n_partitions

    @property
    def n_features(self) -> int:
        return self._n_features

    @property
    def n_targets(self) -> int:
        return self._n_targets

    def get_datapoints(self) -> DatasetPartition:
        return cast("DatasetPartition", self.torch_dataset)

    def get_partitions(self) -> list[DatasetPartition]:
        """
        Return the dataset divided into partitions for distribution among agents.

        This method provides the core partitioning functionality for distributed
        optimization. Each partition represents the local dataset of an agent in
        the network.

        Each partition is sampled uniformly at random from the dataset without replacement
        if heterogeneity is False, otherwise each partition contains unique classes (targets_per_partition)
        with number of datapoints per partition equal to
        min(samples_per_partition, number of available datapoints for the selected classes).

        Returns:
            Sequence[DatasetPartition]: Sequence of DatasetPartition objects, where each partition is a list of
            (features, targets) tuples.

        """
        if self._partitions is None:
            if self.heterogeneity:
                self._partitions = self._heterogeneous_split()
            else:
                self._partitions = self._random_split()

        return self._partitions

    def _random_split(self) -> list[DatasetPartition]:
        if self.samples_per_partition is None:
            parts = [1 / self.n_partitions] * self.n_partitions
        elif self.samples_per_partition * self.n_partitions <= self.n_samples:
            parts = [self.samples_per_partition] * self.n_partitions
            # Add the remaining samples to the last partition and remove it
            # to ensure the sum is equal to the total number of samples for random_split
            parts.append(self.n_samples - sum(parts))
        else:
            raise ValueError(
                f"samples_per_partition ({self.samples_per_partition}) * n_partitions ({self.n_partitions}) "
                f"must be <= n_datapoints ({self.n_samples})"
            )

        generator = None
        if self.seed is not None:
            generator = Generator().manual_seed(self.seed)  # pyright: ignore[reportPossiblyUnboundVariable]

        partitions = cast(
            "list[DatasetPartition]",
            torch_random_split(self.torch_dataset, parts, generator=generator),  # pyright: ignore[reportPossiblyUnboundVariable]
        )

        return partitions[: self.n_partitions]

    def _heterogeneous_split(self) -> list[DatasetPartition]:
        """
        Split dataset so each partition contains unique classes.

        Requires that partitions * classes_per_partition <= classes.
        """
        # Group indices by class in a single pass
        class_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, (_, label) in enumerate(self.torch_dataset):  # type: ignore[misc, arg-type]
            if label in class_to_indices or len(class_to_indices) < (self.n_partitions * self.targets_per_partition):  # type: ignore[has-type]
                class_to_indices[label].append(idx)  # type: ignore[has-type]

        # Set random seed if specified for reproducibility
        if self.seed is not None:
            random.seed(self.seed)

        # Create partitions from class-grouped indices
        idx_partitions = []
        min_n_datapoints = int("inf")
        class_idxs = sorted(class_to_indices.keys())
        # Group classes for each partition
        class_idxs_groups = [
            class_idxs[i : i + self.targets_per_partition]
            for i in range(0, len(class_idxs), self.targets_per_partition)
        ]
        for class_idx_group in class_idxs_groups:
            indices = []
            for class_idx in class_idx_group:
                indices.extend(class_to_indices[class_idx])

            # Shuffle and select subset if needed
            random.shuffle(indices)
            if self.samples_per_partition is not None:
                indices = indices[: self.samples_per_partition]

            min_n_datapoints = min(min_n_datapoints, len(indices))
            idx_partitions.append(indices)

        if self.samples_per_partition is not None and min_n_datapoints < self.samples_per_partition:
            LOGGER.warning(
                f"Warning: Some partitions have less datapoints ({min_n_datapoints}) than "
                f"samples_per_partition ({self.samples_per_partition}) due to class distribution. "
                f"All partitions will be truncated to {min_n_datapoints} datapoints."
            )
        partitions = [TorchSubset(self.torch_dataset, idx[:min_n_datapoints]) for idx in idx_partitions]  # pyright: ignore[reportPossiblyUnboundVariable]

        return cast("list[DatasetPartition]", partitions)

    def __len__(self) -> int:
        return self.n_samples
