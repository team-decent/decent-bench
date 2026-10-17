"""Process-wide benchmark configuration."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from decent_array import interoperability as iop
from decent_array.types import Devices, Frameworks

from decent_bench.utils import _logger

if TYPE_CHECKING:
    from decent_bench.utils._checkpoint_manager import CheckpointManager


@dataclass(frozen=True)
class Config:
    """decent-bench configuration."""

    framework: Frameworks
    device: Devices
    seed: int | None
    log_level: int
    storage_dir: Path | None
    n_checkpoints: int
    compression_level: int


@dataclass
class _ConfigurationState:
    config: Config | None = None
    checkpoint_manager: CheckpointManager | None = None


_STATE = _ConfigurationState()


def configure(
    framework: Frameworks | None = None,
    device: Devices | None = None,
    *,
    seed: int | None = None,
    log_level: int = logging.INFO,
    storage_dir: str | Path | None = None,
    n_checkpoints: int = 3,
    compression_level: int = 1,
) -> None:
    """
    Configure benchmarking set-up.

    This function must be called once before creating benchmark problems, running benchmarks, or computing metrics. If
    the benchmarking stages (problem creation, benchmark, metrics computation and display) are performed in different
    scripts, ``configure`` should be called at the beginning of each.

    When ``storage_dir`` points to an existing experiment, its backend, random seed, and checkpoint settings are used to
    configure the environment. ``framework`` and ``device`` must not be provided in this case.

    Args:
        framework: backend framework to use. Must be provided when configuring a new experiment.
        device: device on which to run the benchmark. Must be provided when configuring a new experiment.
        seed: random seed to use for the benchmark.
        log_level: logging level. See the `logging documentation
            <https://docs.python.org/3/library/logging.html#logging-levels>`_ for available levels.
        storage_dir: directory in which benchmark checkpoints and results are stored. If the directory contains
            an existing experiment, its configuration is loaded from ``metadata.json``.
        n_checkpoints:  number of checkpoints to create for each algorithm and trial. The checkpoints are taken at
            equally spaced intervals throughout each trial, and the last iteration is always included.
        compression_level: compression level used when storing checkpoints. Higher values provide greater compression
            but require more time. See the `Zstandard documentation <https://facebook.github.io/zstd/>`_ for details.

    Raises:
        RuntimeError: if configure has already been called.
        ValueError: if ``storage_dir`` contains an experiment but no metadata, or if framework and device are also
            passed. If ``storage_dir`` is empty but framework and device are not passed.

    """
    if _STATE.config is not None:
        raise RuntimeError("configure() can only be called once per process")

    # extract metadata from existing experiment folder
    storage_path = Path(storage_dir) if storage_dir is not None else None
    metadata: dict[str, Any] | None = None
    if storage_path is not None and storage_path.exists():
        if not storage_path.is_dir():
            raise ValueError(f"Storage path is not a directory: {storage_path}")
        if any(storage_path.iterdir()):
            metadata_path = storage_path / "metadata.json"
            if not metadata_path.is_file():
                raise ValueError(f"Existing storage directory '{storage_path}' does not contain metadata.json")
            with metadata_path.open(encoding="utf-8") as file:
                metadata = json.load(file)
            if framework is not None or device is not None:
                raise ValueError("framework and device must not be provided when opening an existing experiment")

            backend = metadata["backend"]
            framework = Frameworks(backend["framework"])
            device = Devices(backend["device"])
            seed = metadata.get("rng_seed", seed)
            checkpointing = metadata.get("checkpointing", {})
            n_checkpoints = checkpointing.get("n_checkpoints", n_checkpoints)
            compression_level = checkpointing.get("compression_level", compression_level)

    # arg validation for new experiment
    if framework is None or device is None:
        raise ValueError("framework and device must be provided when configuring a new experiment")
    if n_checkpoints <= 0:
        raise ValueError(f"n_checkpoints must be a positive integer, got {n_checkpoints}")

    # apply configuration
    iop.set_backend(framework, device)
    if seed is not None:
        iop.set_seed(seed)
    _logger.start_logger(log_level=log_level)
    _STATE.config = Config(framework, device, seed, log_level, storage_path, n_checkpoints, compression_level)

    # checkpoint manager
    if storage_path is not None:
        from decent_bench.utils._checkpoint_manager import CheckpointManager  # noqa: PLC0415

        _STATE.checkpoint_manager = CheckpointManager(
            storage_path,
            n_checkpoints=n_checkpoints,
            compression_level=compression_level,
        )


def get_config() -> Config:
    """Return the current configuration or fail if ``configure()`` was omitted."""
    if _STATE.config is None:
        raise RuntimeError("configure() must be called before using decent-bench")
    return _STATE.config


def get_checkpoint_manager() -> CheckpointManager | None:
    """Return the configured checkpoint manager."""
    get_config()
    return _STATE.checkpoint_manager
