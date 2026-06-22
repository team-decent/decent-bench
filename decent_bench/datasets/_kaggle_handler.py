from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import DTypeLike

try:
    import kagglehub  # type: ignore[import-untyped]
    import pandas as pd

    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False

import decent_bench.utils.interoperability as iop
from decent_bench.utils.types import Dataset, SupportedDevices, SupportedFrameworks

from ._dataset_handler import DatasetHandler


class KaggleDatasetHandler(DatasetHandler):
    def __init__(
        self,
        kaggle_handle: str,
        path: str,
        feature_columns: list[str],
        target_columns: list[str],
        *,
        framework: SupportedFrameworks = SupportedFrameworks.NUMPY,
        device: SupportedDevices = SupportedDevices.CPU,
        dtype: DTypeLike = np.float64,
        partition_label_column: str | None = None,
    ) -> None:
        """
        Dataset wrapper for Kaggle datasets.

        Args:
            kaggle_handle: Kaggle dataset handle, e.g. "user_name/dataset_name"
            path: Path to the dataset file within the Kaggle dataset
            feature_columns: List of feature column names
            target_columns: List of target column names
            framework: Framework to use for data representation
            device: Device to use for data representation
            dtype: Data type of the returned arrays
            partition_label_column: Column exposed to label-based splitting utilities. Defaults to the only
                target column when exactly one target column is configured.

        Raises:
            ImportError: If kagglehub or pandas is not installed
            RuntimeError: If the dataset fails to load from Kaggle
            ValueError: If the configured partition label column is not found.

        Note:
            If you need to authenticate with Kaggle, ensure that your Kaggle API credentials
            are set up correctly. Easiest solution is to set your api token in the environment variable
            ``KAGGLE_API_TOKEN``. Refer to https://www.kaggle.com/docs/api for more information.

        """
        if not KAGGLE_AVAILABLE:
            raise ImportError(
                "kagglehub and pandas are required to use KaggleDataset. "
                "Install them with: pip install kagglehub pandas"
            )

        self.kaggle_handle = kaggle_handle
        self.path = path
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.partition_label_column = partition_label_column
        self.framework = framework
        self.device = device
        self.dtype = dtype

        self._df: pd.DataFrame = kagglehub.dataset_load(  # pyright: ignore[reportPossiblyUnboundVariable]
            kagglehub.KaggleDatasetAdapter.PANDAS,  # pyright: ignore[reportPossiblyUnboundVariable]
            self.kaggle_handle,
            self.path,
        )
        if self._df is None:
            raise RuntimeError(f"Failed to load dataset from Kaggle handle: {self.kaggle_handle}, path: {self.path}")

        if self.partition_label_column is not None and self.partition_label_column not in self._df.columns:
            raise ValueError(f"partition_label_column ({self.partition_label_column}) is not in the dataframe")

    @property
    def n_samples(self) -> int:
        return len(self._df)

    @property
    def n_features(self) -> int:
        return len(self.feature_columns)

    @property
    def n_targets(self) -> int:
        return len(self.target_columns)

    def get_datapoints(self) -> Dataset:
        return self._create_partition(self._df)

    def get_labels(self) -> list[object]:
        """Return labels from the configured partition label column."""
        label_column = self.partition_label_column
        if label_column is None:
            if len(self.target_columns) != 1:
                raise ValueError("partition_label_column is required when using multiple target columns")
            label_column = self.target_columns[0]
        return list(self._df[label_column])

    def split(
        self,
        partitions: Sequence[Sequence[int]],
    ) -> Sequence[Dataset]:
        """Materialize dataframe rows from index partitions."""
        idx_partitions = self._resolve_partitions(partitions)
        return [self._create_partition(self._df.iloc[indices]) for indices in idx_partitions]

    def _create_partition(self, df_partition: pd.DataFrame) -> Dataset:
        partition: Dataset = []
        for _, row in df_partition.iterrows():
            x = iop.to_array(
                row[self.feature_columns].to_numpy().astype(self.dtype),
                framework=self.framework,
                device=self.device,
            )
            y = iop.to_array(
                row[self.target_columns].to_numpy().astype(self.dtype),
                framework=self.framework,
                device=self.device,
            )
            partition.append((x, y))
        return partition
