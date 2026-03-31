from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from numpy import float64
from numpy.typing import NDArray
from PIL import Image

from decent_bench.datasets import DatasetHandler
from decent_bench.utils.array import Array
from decent_bench.utils.logger import LOGGER
from decent_bench.utils.types import Dataset

from .lidar import (
    compute_headings,
    densify_path,
    image_to_occupancy,
    sample_along_path,
)


class NIMDatasetHandler(DatasetHandler):
    """Neural Implicit Mapping Dataset."""

    def __init__(
        self,
        image_file: str,
        n_partitions: int,
        samples_per_partition: int | None = None,
        transform: Callable[[tuple[int, int]], Array] | None = None,
        label_transform: Callable[[bool], Array] | None = None,
        *,
        # Random sampling options
        leakage: float = 0.0,
        seed: int | None = None,
        balance_labels: bool = False,
        # Path-based lidar sampling options
        paths: list[list[tuple[int, int]]] | str | None = None,
        samples_per_pose: int = 5,
        num_beams: int = 36,
        fov: float = 2 * math.pi,
        max_range: float | None = None,
        scan_spacing: float | None = None,
    ) -> None:
        self.image_file = image_file
        self._n_partitions = n_partitions
        self.samples_per_partition = samples_per_partition
        self.transform = transform
        self.label_transform = label_transform
        # Random sampling options
        self.leakage = leakage
        self.seed = seed
        self.balance_labels = balance_labels
        # Path-based lidar sampling options
        self.paths = paths
        self.samples_per_pose = samples_per_pose
        self.num_beams = num_beams
        self.fov = fov
        self.max_range = max_range
        self.scan_spacing = scan_spacing

        self._n_samples: int | None = None
        # Set random seed for reproducible sampling
        if self.seed is not None:
            random.seed(self.seed)

        # Load and process the image
        image = Image.open(self.image_file).convert("L")
        image_array = np.array(image, dtype=float64) / 255.0
        self.height, self.width = image_array.shape
        self.feature_norm = max(self.height, self.width)

        # If configured to use provided paths for LIDAR sampling, do that and return
        res = (
            self._lidar(image_array)
            if self.paths is not None
            else self._random_sampling(image_array)
        )

        if len(res) != self.n_partitions:
            raise ValueError(
                f"Number of generated partitions {len(res)} does not match the requested {self.n_partitions} partitions"
                f". Check your configuration and input data to ensure it can produce the requested number of "
                f"partitions. If you think this should be possible, please open an issue with details about your "
                f"configuration and input data at the decent-bench Github repository."
            )

        self._partitions = self._normalize_features(res)

    @property
    def n_samples(self) -> int:  # noqa: D102
        if self._n_samples is None:
            raise ValueError(
                "n_samples is not set until get_partitions or get_datapoints is accessed at least once"
            )

        return self._n_samples

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

    def get_datapoints(self) -> Dataset:  # noqa: D102
        return [item for partition in self._partitions for item in partition]

    def get_partitions(self) -> Sequence[Dataset]:  # noqa: D102
        return self._partitions

    def _identity_transform(self, x: object) -> Array:
        return x  # type: ignore[return-value]

    def _lidar(  # noqa: PLR0912
        self,
        image_array: NDArray[float64],
    ) -> Sequence[Dataset]:
        occ = image_to_occupancy(image_array, threshold=0.5)

        # Load paths either from provided list or file
        if isinstance(self.paths, list):
            paths_list = self.paths
        elif self.paths is not None:
            try:
                with Path(self.paths).open("r", encoding="utf-8") as f:
                    paths_list = json.load(f)
            except Exception as e:
                raise RuntimeError(
                    f"Unable to load paths file {self.paths}: {e}"
                ) from e
        else:
            raise ValueError("Paths must be provided as a list or a file path")

        if len(paths_list) < self.n_partitions:
            raise ValueError(
                f"Number of provided paths {len(paths_list)} is less than the number of partitions {self.n_partitions}"
            )

        if len(paths_list) > self.n_partitions:
            LOGGER.warning(
                f"Truncated paths list to {self.n_partitions} partitions from "
                f"{len(paths_list)} available paths."
            )
            paths_list = paths_list[: self.n_partitions]

        res_paths: list[Dataset] = []
        for path in paths_list:
            # densify based on scan_spacing; always compute headings if requested
            positions = (
                densify_path(path, self.scan_spacing) if self.scan_spacing else path
            )
            poses = compute_headings(positions)
            samples = sample_along_path(
                occ,
                poses,
                samples_per_pose=self.samples_per_pose,
                num_beams=self.num_beams,
                fov=self.fov,
                max_range=(
                    max(self.height, self.width) / 2
                    if self.max_range is None
                    else self.max_range
                ),
            )

            # Limit to samples_per_partition if set
            if self.samples_per_partition is not None:
                if len(samples) > self.samples_per_partition:
                    sampled = random.sample(samples, self.samples_per_partition)
                else:
                    sampled = samples
                    LOGGER.warning(
                        f"Partition has {len(samples)} samples, which is less than the requested "
                        f"{self.samples_per_partition}. Using as many samples as available without duplication."
                    )
            else:
                sampled = samples

            # Apply transforms
            if not self.transform:
                self.transform = self._identity_transform
            if not self.label_transform:
                self.label_transform = self._identity_transform

            sampled_partition = [
                (
                    self.transform(s[0]),
                    self.label_transform(s[1]),
                )
                for s in sampled
            ]
            res_paths.append(sampled_partition)

        return res_paths

    def _random_sampling(self, image_array: NDArray[float64]) -> Sequence[Dataset]:
        # Get spatial partitions from the image
        spatial_partitions = self._create_spatial_partitions(image_array)

        res: list[Dataset] = []
        for partition_data in spatial_partitions:
            data = partition_data
            if self.balance_labels:
                # Balance classes in this partition by removing the majority class
                labels = np.array([item[1] for item in data])
                unique, counts = np.unique(labels, return_counts=True)
                class_counts = dict(zip(unique, counts, strict=True))
                min_count = min(class_counts.values())

                balanced_partition = []
                for cls in unique:
                    cls_samples = [item for item in data if item[1] == cls]
                    # Randomly sample from the class to match the minimum count
                    balanced_partition.extend(random.sample(cls_samples, min_count))

                data = balanced_partition

            # Sample from this spatial partition
            if self.samples_per_partition is not None:
                if len(data) > self.samples_per_partition:
                    sampled_partition = random.sample(data, self.samples_per_partition)
                else:
                    sampled_partition = data
                    LOGGER.warning(
                        f"Partition has {len(data)} samples, which is less than the requested "
                        f"{self.samples_per_partition}. Using as many samples as available without duplication."
                    )
            else:
                sampled_partition = data

            if sampled_partition:
                if not self.transform:
                    self.transform = self._identity_transform
                if not self.label_transform:
                    self.label_transform = self._identity_transform

                res.append(
                    [
                        (self.transform(item[0]), self.label_transform(item[1]))
                        for item in sampled_partition
                    ]
                )

        return res

    def _create_spatial_partitions(
        self,
        image_array: NDArray[float64],
    ) -> list[list[tuple[tuple[int, int], bool]]]:
        """
        Divide the image into spatial partitions (squares) and group pixels by partition.

        Raises:
            ValueError: If it's not possible to create a grid with exactly the requested number of partitions.

        """
        # Create coordinate arrays for all pixels
        y_coords, x_coords = np.meshgrid(
            range(self.height), range(self.width), indexing="ij"
        )

        # Create labels: 1 for black (< 0.5), 0 for white (>= 0.5)
        labels = image_to_occupancy(image_array.flatten()).reshape(-1, 1)
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
            int(self.leakage * partition_height),
            int(self.leakage * partition_width),
        )

        # Initialize partition lists
        spatial_partitions: list[list[tuple[tuple[int, int], bool]]] = [
            [] for _ in range(self.n_partitions)
        ]

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
                    spatial_partitions[r * cols + c].append(((x, y), label == 1))

        return spatial_partitions

    def _normalize_features(self, partitions: Sequence[Dataset]) -> list[Dataset]:
        """Normalize features in each partition to [0,1] range based on image norm."""
        normalized_partitions: list[list[tuple[Array, Array]]] = []
        for partition in partitions:
            normalized_data = []
            for feature, label in partition:
                normalized_data.append((feature / self.feature_norm, label))
            normalized_partitions.append(normalized_data)
        return normalized_partitions
