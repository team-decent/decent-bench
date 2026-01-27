from __future__ import annotations

from collections.abc import Sequence

import kagglehub  # type: ignore[import-untyped]
import pandas as pd

import decent_bench.utils.interoperability as iop
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks

from ._dataset import Dataset, DatasetPartition


class KaggleDataset(Dataset):
    def __init__(
        self,
        kaggle_handle: str,
        path: str,
        feature_columns: list[str],
        target_columns: list[str],
        partitions: int,
        *,
        framework: SupportedFrameworks = SupportedFrameworks.NUMPY,
        device: SupportedDevices = SupportedDevices.CPU,
        samples_per_partition: int | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Dataset wrapper for Kaggle datasets.

        Args:
            kaggle_handle: Kaggle dataset handle, e.g. "endofnight17j03/iris-classification"
            path: Path to the dataset file within the Kaggle dataset
            feature_columns: List of feature column names
            target_columns: List of target column names
            partitions: Number of partitions to split the dataset into
            framework: Framework to use for data representation
            device: Device to use for data representation
            samples_per_partition: Number of samples per partition
            seed: Random seed for shuffling the dataset

        Note:
            If you need to authenticate with Kaggle, ensure that your Kaggle API credentials
            are set up correctly. Easiest solution is to set your api token in the environment variable
            ``KAGGLE_API_TOKEN``. Refer to https://www.kaggle.com/docs/api for more information.

        """
        self.kaggle_handle = kaggle_handle
        self.path = path
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.partitions = partitions
        self.framework = framework
        self.device = device
        self.samples_per_partition = samples_per_partition
        self.seed = seed
        self.res: Sequence[DatasetPartition] | None = None

    def training_partitions(self) -> Sequence[DatasetPartition]:
        if self.res is not None:
            return self.res

        df: pd.DataFrame = kagglehub.dataset_load(kagglehub.KaggleDatasetAdapter.PANDAS, self.kaggle_handle, self.path)
        self.res = self._random_split(df)
        return self.res

    def _random_split(self, df: pd.DataFrame) -> Sequence[DatasetPartition]:
        if self.samples_per_partition is None:
            self.samples_per_partition = len(df) // self.partitions

        # Shuffle the dataframe
        df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        partitions: list[DatasetPartition] = []
        for i in range(self.partitions):
            start_idx = i * self.samples_per_partition
            end_idx = start_idx + self.samples_per_partition
            partition_df = df.iloc[start_idx:end_idx]

            partition = []
            for _, row in partition_df.iterrows():
                x = iop.to_array(row[self.feature_columns].to_numpy(), framework=self.framework, device=self.device)
                y = iop.to_array(row[self.target_columns].to_numpy(), framework=self.framework, device=self.device)
                partition.append((x, y))
            partitions.append(partition)

        return partitions
