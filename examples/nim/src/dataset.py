import random
from collections import Counter
from collections.abc import Sequence
from typing import cast

import numpy as np
import torch
from numpy import float64
from numpy.typing import NDArray
from PIL import Image

from decent_bench.datasets import DatasetHandler
from decent_bench.utils.logger import LOGGER, start_logger
from decent_bench.utils.types import Dataset


def image_to_occupancy(
    image_array: NDArray[np.float64],
    threshold: float = 0.8,
) -> NDArray[np.uint8]:
    """
    Convert grayscale image (0..1) to boolean occupancy map.

    True means occupied (wall), False means free.

    Args:
        image_array: 2D array of shape (H,W) with values in [0,1]
        threshold: value below which is considered occupied

    Returns:
        2D boolean array of shape (H,W) where True=occupied, False=free.

    """
    return (image_array < threshold).astype(np.uint8)


class NIMDatasetHandler(DatasetHandler):
    """
    Neural Implicit Mapping Dataset.

    Partition layout:
        The image domain is divided into a regular grid of ``rows x cols`` axis-aligned rectangles.
        The grid is chosen to use exactly ``n_partitions`` cells and to be as close to square as possible.
        Concretely, the code starts with a number of rows near the square-root of ``n_partitions`` and decreases it
        until it evenly divides ``n_partitions``; the number of columns is then whatever is needed to reach exactly
        ``n_partitions``.

        If ``overlap > 0``, partitions overlap slightly: pixels near a partition boundary are also included in the
        neighboring partition(s). The overlap size is derived from the partition height/width.

    Note:
        Coordinates are normalized to [0,1] by dividing by the maximum image dimension (width or height).
        The normalization factor is determined from the original image dimensions before
        partitioning, so all partitions are normalized consistently based on the same factor (:attr:`feature_norm`).

    Args:
        image_file: Path to the input image file (grayscale, where black above occupancy_threshold=occupied and any
            pixel value below occupancy_threshold=free).
        n_partitions: Number of spatial partitions to divide the data into.
        samples_per_partition: Optional max number of samples per partition after balancing. If None, keep all samples.
        label_balance: Optional ratio of majority to minority class in each partition after balancing. Attempts to
            balance labels as closely to 1:1 as possible, with a maximum ratio of 1:label_balance. If None, no
            balancing is done. Must be >= 1 if set.
        occupancy_threshold: Threshold for converting grayscale values to occupancy labels. Defaults to 0.8.
        overlap: Fraction of partition size to allow for overlapping samples between partitions.
            Defaults to 0 (no overlap).

    """

    def __init__(
        self,
        image_file: str,
        n_partitions: int,
        samples_per_partition: int | None = None,
        *,
        label_balance: float | None = None,
        occupancy_threshold: float = 0.8,
        overlap: float = 0.0,
    ) -> None:
        start_logger()
        self.image_file = image_file
        self._n_partitions = n_partitions
        self.samples_per_partition = samples_per_partition
        self.overlap = overlap
        self.label_balance = label_balance
        self.occupancy_threshold = occupancy_threshold

        if self.label_balance is not None and self.label_balance < 1:
            raise ValueError("label_balance must be >= 1")

        # Load and process the image
        image = Image.open(self.image_file).convert("L")
        image_array = np.array(image, dtype=float64) / 255.0
        self.height, self.width = image_array.shape
        self._feature_norm: float = max(self.height, self.width)

        res = self._create_spatial_partitions(image_array)

        if len(res) != self.n_partitions:
            raise ValueError(
                f"Number of generated partitions {len(res)} does not match the requested {self.n_partitions} partitions"
                f". Check your configuration and input data to ensure it can produce the requested number of "
                f"partitions. If you think this should be possible, please open an issue with details about your "
                f"configuration and input data at the decent-bench Github repository."
            )

        res = self._balance_partitions(
            res,
            label_balance=self.label_balance,
            samples_per_partition=self.samples_per_partition,
        )
        self._partitions = self._apply_transforms_and_normalize(res)

    @property
    def n_samples(self) -> int:  # noqa: D102
        return len(self.get_datapoints())

    @property
    def n_partitions(self) -> int:  # noqa: D102
        return self._n_partitions

    @property
    def n_features(self) -> int:  # noqa: D102
        return 2  # (x, y) coordinates

    @property
    def n_targets(self) -> int:
        """
        Number of target dimensions.

        1 for occupancy label (0 or 1) in the NIM dataset,
        can bee seen as the density value that the model is trying to predict at each (x, y) coordinate.
        """
        return 1  # occupancy label (0 or 1)

    @property
    def feature_norm(self) -> float:
        """Normalization factor for features (coordinates)."""
        return self._feature_norm

    def get_datapoints(self) -> Dataset:  # noqa: D102
        return [item for partition in self._partitions for item in partition]

    def get_partitions(self) -> Sequence[Dataset]:  # noqa: D102
        return self._partitions

    def get_test_set(self, label_balance: float | None = None, num_samples: int | None = None) -> Dataset:
        """Return occupancy grid as the test set."""
        image = Image.open(self.image_file).convert("L")
        image_array = np.array(image, dtype=float64) / 255.0
        occ = image_to_occupancy(image_array, threshold=self.occupancy_threshold)
        y_coords, x_coords = np.meshgrid(range(self.height), range(self.width), indexing="ij")
        features = np.column_stack([x_coords.flatten(), y_coords.flatten()]).astype(int)
        labels = occ.flatten().astype(int)

        raw_test_set: list[tuple[tuple[int, int], int]] = []
        for (x, y), label in zip(features, labels, strict=True):
            raw_test_set.append(((int(x), int(y)), int(label)))

        raw_test_set = self._balance_partitions(
            [raw_test_set],
            label_balance=label_balance,
            samples_per_partition=num_samples,
        )[0]
        return self._apply_transforms_and_normalize([raw_test_set])[0]

    def _feature_transform(self, x: tuple[int, int]) -> torch.Tensor:
        return torch.tensor(x)

    def _target_transform(self, x: int) -> torch.Tensor:
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0)

    def _create_spatial_partitions(
        self,
        image_array: NDArray[float64],
    ) -> list[list[tuple[tuple[int, int], int]]]:
        """
        Divide the image into spatial partitions (squares) and group pixels by partition.

        Raises:
            ValueError: If it's not possible to create a grid with exactly the requested number of partitions.

        """
        # Create coordinate arrays for all pixels
        y_coords, x_coords = np.meshgrid(range(self.height), range(self.width), indexing="ij")

        # Create labels: 1 for black (< 0.5), 0 for white (>= 0.5)
        labels = image_to_occupancy(image_array.flatten(), threshold=self.occupancy_threshold).reshape(-1, 1)
        features = np.column_stack([x_coords.flatten(), y_coords.flatten()]).astype(int)

        # Find grid dimensions that give exactly the requested number of partitions
        # Try to find the most "square-like" grid that has exactly the requested partitions
        rows = int(np.floor(np.sqrt(self.n_partitions)))
        while self.n_partitions % rows != 0 and rows > 1:
            rows -= 1
        cols = self.n_partitions // rows

        # Double check we have exactly the right number of partitions
        if rows * cols != self.n_partitions:
            raise ValueError(
                f"Unable to create a grid with exactly {self.n_partitions} partitions. "
                f"Closest we can get is {rows} rows and {cols} columns for a total of {rows * cols} partitions. "
                f"Please adjust the number of partitions or the image dimensions to allow for an exact grid."
            )

        # Calculate partition dimensions
        partition_height = self.height // rows
        partition_width = self.width // cols

        # Calculate leakage in pixels based on partition dimensions
        leakage = min(
            int(self.overlap * partition_height),
            int(self.overlap * partition_width),
        )

        # Initialize partition lists
        spatial_partitions: list[list[tuple[tuple[int, int], int]]] = [[] for _ in range(self.n_partitions)]

        # Group pixels by their spatial partition
        for (x, y), label in zip(features, labels, strict=True):
            # Find all potential partitions this pixel could belong to based on leakage
            # Calculate row/column ranges that this pixel might belong to
            min_row = max(0, int((y - leakage) // partition_height))
            max_row = min(rows - 1, int((y + leakage) // partition_height))
            min_col = max(0, int((x - leakage) // partition_width))
            max_col = min(cols - 1, int((x + leakage) // partition_width))

            # Add the pixel to all partitions it belongs to
            for r in range(min_row, max_row + 1):
                for c in range(min_col, max_col + 1):
                    spatial_partitions[r * cols + c].append(((int(x), int(y)), label.item()))

        return spatial_partitions

    def _balance_partitions(  # noqa: PLR0912
        self,
        partitions: list[list[tuple[tuple[int, int], int]]],
        label_balance: float | None = None,
        samples_per_partition: int | None = None,
    ) -> list[list[tuple[tuple[int, int], int]]]:
        if label_balance is None and samples_per_partition is None:
            return partitions  # No balancing needed

        balanced_partitions: list[list[tuple[tuple[int, int], int]]] = []
        for partition in partitions:
            if not partition:
                balanced_partitions.append([])
                continue

            # Only want to have samples_per_partition but not balanced
            if label_balance is None and samples_per_partition is not None:
                # If only samples_per_partition is set, sample randomly from the partition without balancing
                balanced_partition = random.sample(partition, min(samples_per_partition, len(partition)))
                balanced_partitions.append(balanced_partition)
                continue

            labels = [item[1] for item in partition]
            class_counts = dict(Counter(labels))
            min_count = min(class_counts.values())

            if len(class_counts) < 2:
                # If there's only one class present, we can't balance, so we just sample if needed
                if samples_per_partition is not None and len(partition) > samples_per_partition:
                    balanced_partition = random.sample(partition, samples_per_partition)
                else:
                    balanced_partition = partition
                balanced_partitions.append(balanced_partition)
                LOGGER.warning(
                    f"Partition has only one class present. Unable to balance labels. "
                    f"{len(partition)} samples in this partition, class distribution: {class_counts}."
                )
                continue

            LOGGER.info(
                f"Balancing partition with {len(partition)} samples, class distribution: "
                f"{sorted(class_counts.items())}, min class count: {min_count}, label_balance: {label_balance}, "
                f"samples_per_partition: {samples_per_partition}."
            )

            desired_counts: dict[int, int] = {}
            if label_balance is not None and samples_per_partition is None:
                # If only label_balance is set, calculate samples_per_partition
                # based on the minority class and balance ratio
                desired_counts[1] = int(min_count * label_balance)
                desired_counts[0] = int(min_count * label_balance)
            elif label_balance is not None and samples_per_partition is not None:
                # If both label_balance and samples_per_partition are set, calculate the number
                # of samples for each class based on the balance ratio and total samples per partition
                if min_count >= samples_per_partition // 2:
                    desired_counts[1] = samples_per_partition // 2
                    desired_counts[0] = samples_per_partition // 2
                else:
                    minority_label = min(class_counts, key=lambda label: class_counts[label])
                    majority_label = max(class_counts, key=lambda label: class_counts[label])
                    desired_counts[minority_label] = class_counts[minority_label]
                    desired_counts[majority_label] = min(
                        int(class_counts[minority_label] * label_balance),
                        samples_per_partition - class_counts[minority_label],
                    )

            balanced_partition = []
            for label, count in class_counts.items():
                if count > desired_counts[label]:
                    # If there are more samples than the max allowed for this class, sample down to the max allowed
                    label_partition = [item for item in partition if item[1] == label]
                    subset = random.sample(label_partition, desired_counts[label])
                else:
                    # If there are fewer samples than the max allowed, keep all samples for this class
                    subset = [item for item in partition if item[1] == label]
                balanced_partition.extend(subset)

            balanced_partitions.append(balanced_partition)

            LOGGER.info(
                f"After balancing, partition has {len(balanced_partition)} samples, class distribution: "
                f"{sorted(dict(Counter([item[1] for item in balanced_partition])).items())}."
            )

        return balanced_partitions

    def _apply_transforms_and_normalize(self, partitions: list[list[tuple[tuple[int, int], int]]]) -> list[Dataset]:
        """Normalize features in each partition to [0,1] range based on image norm."""
        transformed_partitions: list[Dataset] = []
        for partition in partitions:
            transformed_partition = []
            for feature, label in partition:
                transformed_partition.append((
                    self._feature_transform(feature) / self._feature_norm,
                    self._target_transform(label),
                ))
            transformed_partitions.append(cast("Dataset", transformed_partition))
        return transformed_partitions
