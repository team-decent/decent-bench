from __future__ import annotations

from collections import defaultdict
import random
from collections.abc import Sequence
from typing import Any, cast

from torch import Generator
from torch import Tensor as TorchTensor
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Subset as TorchSubset
from torch.utils.data import random_split as torch_random_split

from ._dataset import Dataset, DatasetPartition


class PyTorchWrapper(Dataset):
    def __init__(
        self,
        torch_dataset: TorchDataset[Any],
        features: int,
        targets: int,
        partitions: int,
        *,
        samples_per_partition: int | None = None,
        heterogeneity: bool = False,
        targets_per_partition: int = 1,
        seed: int | None = None,
    ) -> None:
        """
        Dataset wrapper for PyTorch datasets which represents datapoints as tuples (features, targets).

        This class can create either random partitions or heterogeneous partitions where each partition
        contains unique classes. This class will preserve the properties of the underlying PyTorch dataset
        such as transforms and lazy-loading.

        Args:
            torch_dataset: PyTorch dataset to wrap
            features: Number of feature dimensions
            targets: Number of target dimensions
            partitions: Number of partitions to split the dataset into
            samples_per_partition: Number of samples per partition, if None, will split evenly
            heterogeneity: Whether to create heterogeneous partitions with unique classes
            targets_per_partition: Number of unique classes per partition (only if heterogeneity is True)
            seed: Random seed for shuffling the dataset

        Raises:
            ValueError: If heterogeneity is True and partitions * targets_per_partition > targets

        Note:
            If heterogeneity is True, each partition will contain unique classes.
            Ensure that partitions * targets_per_partition <= targets. Be aware that
            this may lead to some classes being unused if the condition is not tight
            and the self.targets attribute will be updated accordingly.

        """
        self.torch_dataset = torch_dataset
        self.targets = targets
        self.features = features
        self.partitions = partitions
        self.samples_per_partition = samples_per_partition
        self.heterogeneity = heterogeneity
        self.targets_per_partition = targets_per_partition
        self.seed = seed

        if self.heterogeneity:
            if (self.partitions * self.targets_per_partition) > self.targets:
                raise ValueError(
                    f"Number of partitions ({self.partitions}) * targets per partition ({self.targets_per_partition})"
                    f" must be <= targets ({self.targets})"
                )
            # Set the new number of used targets
            self.targets = self.partitions * self.targets_per_partition
        elif self.samples_per_partition is None:
            if not hasattr(self.torch_dataset, "__len__"):
                raise ValueError(
                    "samples_per_partition must be set if the dataset length is not known, len() not implemented."
                )
            self.dataset_len = len(self.torch_dataset)
            self.samples_per_partition = self.dataset_len // self.partitions

    def training_partitions(self) -> Sequence[DatasetPartition]:
        if self.heterogeneity:
            return self._heterogeneous_split()

        return self._random_split()

    def _random_split(self) -> Sequence[DatasetPartition]:
        if self.samples_per_partition is None:
            parts = [1 / self.partitions] * self.partitions
        else:
            parts = [self.samples_per_partition] * self.partitions
            # Add the remaining samples to the last partition and remove it
            # to ensure the sum is equal to the total number of samples for random_split
            parts.append(self.dataset_len - sum(parts))

        generator = None
        if self.seed is not None:
            generator = Generator().manual_seed(self.seed)

        partitions = cast(
            "Sequence[DatasetPartition]",
            torch_random_split(self.torch_dataset, parts, generator=generator),
        )

        return partitions[: self.partitions]

    def _heterogeneous_split(self) -> Sequence[DatasetPartition]:
        """
        Split dataset so each partition contains unique classes.

        Requires that partitions * classes_per_partition <= classes.
        """
        # Group indices by class in a single pass
        class_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, (_, label) in enumerate(self.torch_dataset):  # type: ignore[misc, arg-type]
            if label in class_to_indices or len(class_to_indices) < (self.partitions * self.targets_per_partition):  # type: ignore[has-type]
                class_to_indices[label].append(idx)  # type: ignore[has-type]

        # Set random seed if specified for reproducibility
        if self.seed is not None:
            random.seed(self.seed)

        # Create partitions from class-grouped indices
        partitions = []
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

            partitions.append(TorchSubset(self.torch_dataset, indices))

        return cast("Sequence[DatasetPartition]", partitions)
