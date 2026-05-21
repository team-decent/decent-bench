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
from ._partitioners import IidPartitioner, Partitioner


class KaggleDatasetHandler(DatasetHandler):
    def __init__(
        self,
        kaggle_handle: str,
        path: str,
        feature_columns: list[str],
        target_columns: list[str],
        n_partitions: int | None = None,
        *,
        framework: SupportedFrameworks = SupportedFrameworks.NUMPY,
        device: SupportedDevices = SupportedDevices.CPU,
        dtype: DTypeLike = np.float64,
        samples_per_partition: int | None = None,
        partitioner: Partitioner | None = None,
        partition_label_column: str | None = None,
    ) -> None:
        """
        Dataset wrapper for Kaggle datasets.

        Args:
            kaggle_handle: Kaggle dataset handle, e.g. "user_name/dataset_name"
            path: Path to the dataset file within the Kaggle dataset
            feature_columns: List of feature column names
            target_columns: List of target column names
            n_partitions: Number of partitions for the default IID partitioner. Do not use together with partitioner.
            framework: Framework to use for data representation
            device: Device to use for data representation
            dtype: Data type of the returned arrays
            samples_per_partition: Number of samples per partition for the default IID partitioner.
            partitioner: Optional partitioner defining the row split. If omitted, an IID partitioner is created
                from n_partitions and samples_per_partition.
            partition_label_column: Column used by label-based partitioners. Defaults to the only
                target column when exactly one target column is configured.

        Raises:
            ImportError: If kagglehub or pandas is not installed
            RuntimeError: If the dataset fails to load from Kaggle
            ValueError: If there are not enough samples to create the requested partitions

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
        if partitioner is not None and (n_partitions is not None or samples_per_partition is not None):
            raise ValueError("partitioner cannot be combined with n_partitions or samples_per_partition")

        self.kaggle_handle = kaggle_handle
        self.path = path
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.partition_label_column = partition_label_column
        self.framework = framework
        self.device = device
        self.dtype = dtype
        self._partitions: Sequence[Dataset] | None = None

        self._df: pd.DataFrame = kagglehub.dataset_load(  # pyright: ignore[reportPossiblyUnboundVariable]
            kagglehub.KaggleDatasetAdapter.PANDAS,  # pyright: ignore[reportPossiblyUnboundVariable]
            self.kaggle_handle,
            self.path,
        )
        if self._df is None:
            raise RuntimeError(f"Failed to load dataset from Kaggle handle: {self.kaggle_handle}, path: {self.path}")

        if partitioner is None:
            n_partitions = 1 if n_partitions is None else n_partitions
            samples_per_partition = (
                len(self._df) // n_partitions if samples_per_partition is None else samples_per_partition
            )
            partitioner = IidPartitioner(n_partitions=n_partitions, samples_per_partition=samples_per_partition)
        self.partitioner = partitioner

        if self.partitioner is not None and self.partitioner.requires_labels:
            if self.partition_label_column is None:
                if len(self.target_columns) != 1:
                    raise ValueError("partition_label_column is required when using multiple target columns")
                self.partition_label_column = self.target_columns[0]
            if self.partition_label_column not in self._df.columns:
                raise ValueError(f"partition_label_column ({self.partition_label_column}) is not in the dataframe")

    @property
    def n_samples(self) -> int:
        return len(self._df)

    @property
    def n_partitions(self) -> int:
        return self.partitioner.n_partitions

    @property
    def n_features(self) -> int:
        return len(self.feature_columns)

    @property
    def n_targets(self) -> int:
        return len(self.target_columns)

    def get_datapoints(self) -> Dataset:
        return self._create_partition(self._df)

    def get_partitions(self) -> Sequence[Dataset]:
        """
        Return the dataset divided into partitions for distribution among agents.

        This method provides the core partitioning functionality for decentralized
        optimization. Each partition represents the local dataset of an agent in
        the network.

        Each partition is sampled uniformly at random from the dataset without replacement.

        Returns:
            Sequence[Dataset]: Sequence of Dataset objects, where each partition is a list of
            (features, targets) tuples.

        """
        if self._partitions is None:
            self._partitions = self._partitioner_split(self._df)
        return self._partitions

    def _partitioner_split(self, df: pd.DataFrame) -> Sequence[Dataset]:
        labels = None
        if self.partitioner.requires_labels:
            if self.partition_label_column is None:
                raise RuntimeError("partition_label_column is not set")
            labels = list(df[self.partition_label_column])

        idx_partitions = self.partitioner.partition(len(df), labels=labels)
        return [self._create_partition(df.iloc[indices]) for indices in idx_partitions]

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
