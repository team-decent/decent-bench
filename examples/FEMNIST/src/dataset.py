from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import torch

from decent_bench.datasets import DatasetHandler
from decent_bench.utils.types import Dataset

SplitName = Literal["train", "test"]
ImageLayout = Literal["flat", "cnn"]


def load_femnist_metadata(
    dataset_name: str,
    cache_dir: Path | None,
) -> pd.DataFrame:
    """Load lightweight writer and label metadata from the Hugging Face FEMNIST dataset."""
    from datasets import Image, load_dataset  # type: ignore[import-untyped]  # noqa: PLC0415

    dataset = load_dataset(
        dataset_name,
        split="train",
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    if "image" in dataset.features:
        dataset = dataset.cast_column("image", Image(decode=False))

    if not {"writer_id", "character"}.issubset(dataset.column_names):
        raise ValueError(f"Unexpected Hugging Face FEMNIST columns: {dataset.column_names}")

    df = cast("pd.DataFrame", dataset.select_columns(["writer_id", "character"]).to_pandas())
    df.insert(0, "row_index", np.arange(len(df), dtype=np.int64))
    df["writer_id"] = df["writer_id"].astype(str)
    df["label"] = df["character"].astype(int)
    return df[["row_index", "writer_id", "label"]]


def add_seeded_train_test_split(df: pd.DataFrame, train_fraction: float, seed: int) -> pd.DataFrame:
    """Add deterministic per-writer train/test labels."""
    if not 0 < train_fraction < 1:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")

    split_df = df.copy()
    rng = np.random.default_rng(seed)
    split_values = np.empty(len(split_df), dtype=object)

    for _, group in split_df.groupby("writer_id", sort=True):
        positions = group.index.to_numpy()
        shuffled = positions.copy()
        rng.shuffle(shuffled)
        n_train = round(len(shuffled) * train_fraction)
        if len(shuffled) > 1:
            n_train = min(max(n_train, 1), len(shuffled) - 1)
        split_values[shuffled[:n_train]] = "train"
        split_values[shuffled[n_train:]] = "test"

    split_df["split"] = split_values
    return split_df


def client_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-writer train and test sample counts."""
    stats = df.groupby("writer_id", sort=True).agg(
        total_samples=("label", "size"),
        n_classes=("label", "nunique"),
    )
    split_counts = df.pivot_table(
        index="writer_id",
        columns="split",
        values="label",
        aggfunc="size",
        fill_value=0,
    ).rename(columns={"train": "train_samples", "test": "test_samples"})
    for column in ("train_samples", "test_samples"):
        if column not in split_counts:
            split_counts[column] = 0

    stats = stats.join(split_counts[["train_samples", "test_samples"]], how="left").fillna(0)
    stats["train_samples"] = stats["train_samples"].astype(int)
    stats["test_samples"] = stats["test_samples"].astype(int)
    return stats.reset_index()


def choose_writer_ids(
    stats: pd.DataFrame,
    *,
    n_clients: int,
    min_train_samples: int,
    min_test_samples: int,
    seed: int,
) -> list[str]:
    """Select eligible FEMNIST writers deterministically."""
    eligible = stats[(stats["train_samples"] >= min_train_samples) & (stats["test_samples"] >= min_test_samples)]
    if len(eligible) < n_clients:
        raise ValueError(
            f"Requested {n_clients} clients, but only {len(eligible)} satisfy "
            f"min_train_samples={min_train_samples} and min_test_samples={min_test_samples}."
        )

    rng = np.random.default_rng(seed)
    selected_indices = rng.choice(eligible.index.to_numpy(), size=n_clients, replace=False)
    selected = eligible.loc[selected_indices].sort_values("writer_id")
    return selected["writer_id"].astype(str).tolist()


def image_to_tensor(image: object, *, layout: ImageLayout) -> torch.Tensor:
    """Convert a FEMNIST image to a normalized torch tensor."""
    if hasattr(image, "convert"):
        array = np.asarray(image.convert("L"), dtype=np.float32)
    else:
        array = np.asarray(image, dtype=np.float32)

    if array.ndim == 3:
        array = array[..., 0]

    tensor = torch.from_numpy(array / 255.0).to(dtype=torch.float32)
    if layout == "flat":
        return tensor.reshape(-1)
    if layout == "cnn":
        return tensor.unsqueeze(0)
    raise ValueError(f"image_layout must be 'flat' or 'cnn', got {layout!r}")


class FEMNISTDatasetHandler(DatasetHandler):
    """
    FEMNIST handler using natural writer/client partitions.

    FEMNIST is part of the LEAF federated benchmark (Caldas et al., 2018).
    Original source: https://github.com/TalwalkarLab/leaf
    Data is loaded from the ``flwrlabs/femnist`` Hugging Face dataset, which mirrors the LEAF partitioning.
    """

    def __init__(
        self,
        *,
        split: SplitName,
        dataset_name: str = "flwrlabs/femnist",
        cache_dir: Path | str | None = Path("data/femnist/cache"),
        n_clients: int = 10,
        train_fraction: float = 0.8,
        seed: int = 20260524,
        min_train_samples: int = 50,
        min_test_samples: int = 10,
        image_layout: ImageLayout = "flat",
        max_samples_per_client: int | None = 50,
    ) -> None:
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        if n_clients <= 0:
            raise ValueError(f"n_clients must be positive, got {n_clients}")
        if max_samples_per_client is not None and max_samples_per_client <= 0:
            raise ValueError(f"max_samples_per_client must be positive, got {max_samples_per_client}")

        self.split = split
        self.dataset_name = dataset_name
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.requested_n_clients = n_clients
        self.train_fraction = train_fraction
        self.seed = seed
        self.min_train_samples = min_train_samples
        self.min_test_samples = min_test_samples
        self.image_layout = image_layout
        self.max_samples_per_client = max_samples_per_client
        self._partitions: list[Dataset] | None = None

    @property
    def n_samples(self) -> int:  # noqa: D102
        return sum(len(partition) for partition in self.get_partitions())

    @property
    def n_partitions(self) -> int:  # noqa: D102
        return len(self.selected_writer_ids)

    @property
    def n_features(self) -> int:  # noqa: D102
        return 28 * 28

    @property
    def n_targets(self) -> int:  # noqa: D102
        return 62

    @cached_property
    def selected_writer_ids(self) -> list[str]:
        """Return the deterministic selected writer IDs."""
        return choose_writer_ids(
            client_stats(self.metadata),
            n_clients=self.requested_n_clients,
            min_train_samples=self.min_train_samples,
            min_test_samples=self.min_test_samples,
            seed=self.seed,
        )

    @cached_property
    def metadata(self) -> pd.DataFrame:
        """Return FEMNIST metadata with deterministic train/test labels."""
        metadata = load_femnist_metadata(self.dataset_name, self.cache_dir)
        return add_seeded_train_test_split(metadata, train_fraction=self.train_fraction, seed=self.seed)

    @cached_property
    def hf_dataset(self) -> Any:  # noqa: ANN401
        """Return the underlying Hugging Face dataset."""
        from datasets import load_dataset  # type: ignore[import-untyped]  # noqa: PLC0415

        return load_dataset(
            self.dataset_name,
            split="train",
            cache_dir=str(self.cache_dir) if self.cache_dir is not None else None,
        )

    def get_datapoints(self) -> Dataset:  # noqa: D102
        return [datapoint for partition in self.get_partitions() for datapoint in partition]

    def get_partitions(self) -> Sequence[Dataset]:  # noqa: D102
        if self._partitions is None:
            self._partitions = [self._build_writer_partition(writer_id) for writer_id in self.selected_writer_ids]
        return self._partitions

    def _build_writer_partition(self, writer_id: str) -> Dataset:
        writer_rows = self._split_rows_for_writer(writer_id)
        return [self._row_to_datapoint(int(row_index)) for row_index in writer_rows["row_index"].to_list()]

    def _split_rows_for_writer(self, writer_id: str) -> pd.DataFrame:
        writer_rows = self.metadata[
            (self.metadata["writer_id"] == writer_id) & (self.metadata["split"] == self.split)
        ].sort_values("row_index")

        if self.max_samples_per_client is not None:
            writer_rows = writer_rows.head(self.max_samples_per_client)

        return writer_rows

    def _row_to_datapoint(self, row_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.hf_dataset[row_index]
        image = image_to_tensor(row["image"], layout=self.image_layout)
        character = cast("int | str", row["character"])
        label = torch.tensor(int(character), dtype=torch.long)
        return image, label
