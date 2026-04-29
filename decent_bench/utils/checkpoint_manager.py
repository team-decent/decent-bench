import json
import pickle  # noqa: S403
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict, cast

import zstandard as zstd
from rich.progress import track
from rich.status import Status

import decent_bench.utils.interoperability as iop
from decent_bench.agents import AgentMetricsView
from decent_bench.algorithms import Algorithm
from decent_bench.benchmark import BenchmarkProblem, BenchmarkResult, MetricResult
from decent_bench.networks import Network
from decent_bench.utils.logger import LOGGER

# NOTE: On some platforms (notably Windows), the Python binding can expose the 32-bit
# Zstandard magic number as a signed int, which makes it negative.
# `int.to_bytes(..., signed=False)` then raises: OverflowError: can't convert negative int to unsigned
_ZSTD_MAGIC = (int(zstd.MAGIC_NUMBER) & 0xFFFFFFFF).to_bytes(4, "little")
_CHECKPOINT_NAME_RE = re.compile(r"^checkpoint_(\d+)\.pkl(?:\.zst)?$")


class _CheckpointData(TypedDict):
    """Serialized checkpoint payload persisted in each iteration checkpoint file."""

    algorithm: Algorithm[Network]
    network: Network
    iteration: int
    rng_state: dict[str, Any]


class CheckpointManager:  # noqa: PLR0904
    """
    Manages checkpoint directory structure and file operations for benchmark execution.

    The CheckpointManager creates and maintains a hierarchical directory structure for storing
    checkpoint data during benchmark execution. This allows benchmarks to be resumed if interrupted,
    and provides incremental saving of results as trials complete.

    Directory Structure:
        The checkpoint directory is organized as follows::

            checkpoint_dir/
            ├── metadata.json                   # Run configuration and algorithm metadata
            ├── benchmark_problem.pkl.zst       # Initial benchmark problem state (before any trials), zstd-compressed
            ├── initial_algorithms.pkl.zst      # Initial algorithm states (before any trials), zstd-compressed
            ├── metric_computation.pkl.zst      # Computed metrics results (after all trials complete), zstd-compressed
            ├── algorithm_0/                    # Directory for first algorithm
            │   ├── trial_0/                    # Directory for trial 0
            │   │   ├── checkpoint_0000100.pkl.zst  # Combined algorithm+network state at iteration 100, zstd-compressed
            │   │   ├── checkpoint_0000200.pkl.zst  # Combined algorithm+network state at iteration 200, zstd-compressed
            │   │   ├── progress.json           # {"last_completed_iteration": N}
            │   │   └── complete.json           # Marker file, contains path to final checkpoint
            │   ├── trial_1/
            │   │   └── ...
            │   └── trial_N/
            │       └── ...
            └── results/                        # Results directory for storing final tables and plots after completion
                ├── plots_fig1.png              # Final plot for figure 1 with plot results
                ├── plots_fig2.png              # Final plot for figure 2 with plot results
                ├── table.tex                   # Final LaTeX file with table results
                └── table.txt                   # Final text file with table results

    File Descriptions:
        - **metadata.json**: Benchmark configuration and any user-provided metadata
            (e.g., hyperparameters, system info). User-provided metadata can be added through the
            :func:`~decent_bench.benchmark.benchmark` function or appended later using
            :func:`~decent_bench.utils.checkpoint_manager.CheckpointManager.append_metadata`.
        - **benchmark_problem.pkl.zst**: Initial benchmark problem state before any trials run,
            stored as a zstd-compressed pickle payload.
        - **initial_algorithms.pkl.zst**: Initial algorithm states before any trials run,
            stored as a zstd-compressed pickle payload.
        - **metric_computation.pkl.zst**: Computed metrics results after
            :func:`~decent_bench.benchmark.compute_metrics` completes, stored as a
            zstd-compressed pickle payload.
        - **checkpoint_NNNNNNN.pkl.zst**: Combined checkpoint containing both algorithm and network
            state, stored as a zstd-compressed pickle payload. This preserves shared object references and ensures
            consistency between algorithm and network states at each checkpoint. The checkpoint data is a dictionary
            with the following structure:

            - algorithm: :class:`~decent_bench.algorithms.Algorithm`
            - network: :class:`~decent_bench.networks.Network`
            - iteration: iteration

            where "algorithm" is the :class:`~decent_bench.algorithms.Algorithm` object with its internal
            state at the checkpoint, "network" is the :class:`~decent_bench.networks.Network` object with agent states
            at the checkpoint and "iteration" is the iteration number of the checkpoint.
        - **progress.json**: Tracks the last completed iteration within a trial.
        - **complete.json**: Marker file, contains path to final checkpoint.
        - **plots_figX.png**: Final plots for figures after benchmark completion.
        - **table.tex**: Final LaTeX file with table results after benchmark completion.
        - **table.txt**: Final text file with table results after benchmark completion.

    Thread Safety:
        - Each trial writes to its own directory, avoiding write conflicts.
        - Completed trial results are loaded read-only.
        - Metadata is written once at initialization.

    Args:
        checkpoint_dir: Path to save checkpoints during execution. If provided, progress will be saved
            at regular intervals allowing resumption if interrupted. When starting a new benchmark
            the directory must be empty or non-existent.
        checkpoint_step: Number of iterations between checkpoints within each trial.
            If ``None``, only save at the end of each trial. For long-running algorithms,
            set this to checkpoint during trial execution (e.g., every 1000 iterations).
        keep_n_checkpoints: Maximum number of iteration checkpoints to keep per trial.
            Older checkpoints are automatically deleted to save disk space.
        benchmark_metadata: Optional dictionary of additional metadata to save in the checkpoint directory,
                such as hyperparameters or system information. This can be useful for keeping track of the benchmark
                configuration and context when analyzing results later.
        compression_level: Level of compression to use for checkpoint files. Higher levels result in smaller file
            sizes but take more time to compress and decompress. See zstandard documentation
            (:class:`~zstandard.ZstdCompressor`) for details on compression levels. Default is 1, which provides a good
            balance between compression ratio and speed for typical checkpoint payloads. Adjust as needed based on
            the size of the checkpoint data and performance requirements.

    Raises:
        ValueError: If checkpoint_step is not a positive integer or ``None``.
        ValueError: If keep_n_checkpoints is not a positive integer.

    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        checkpoint_step: int | None = None,
        keep_n_checkpoints: int = 3,
        benchmark_metadata: dict[str, Any] | None = None,
        compression_level: int = 1,
    ) -> None:
        """
        Initialize CheckpointManager with a checkpoint directory path.

        Args:
            checkpoint_dir: Path to save checkpoints during execution. If provided, progress will be saved
                at regular intervals allowing resumption if interrupted. When starting a new benchmark
                the directory must be empty or non-existent.
            checkpoint_step: Number of iterations between checkpoints within each trial.
                If ``None``, only save at the end of each trial. For long-running algorithms,
                set this to checkpoint during trial execution (e.g., every 1000 iterations).
            keep_n_checkpoints: Maximum number of iteration checkpoints to keep per trial.
                Older checkpoints are automatically deleted to save disk space.
            benchmark_metadata: Optional dictionary of additional metadata to save in the checkpoint directory,
                    such as hyperparameters or system information. This can be useful for keeping track of the benchmark
                    configuration and context when analyzing results later.
            compression_level: Level of compression to use for checkpoint files. Higher levels result in smaller file
                sizes but take more time to compress and decompress. See zstandard documentation
                (:class:`~zstandard.ZstdCompressor`) for details on compression levels. Default is 1, which provides a
                good balance between compression ratio and speed for typical checkpoint payloads. Adjust as needed based
                on the size of the checkpoint data and performance requirements.

        Raises:
            ValueError: If checkpoint_step is not a positive integer or ``None``.
            ValueError: If keep_n_checkpoints is not a positive integer.

        """
        if checkpoint_step is not None and checkpoint_step <= 0:
            raise ValueError(f"checkpoint_step must be a positive integer or None, got {checkpoint_step}")
        if keep_n_checkpoints <= 0:
            raise ValueError(f"keep_n_checkpoints must be a positive integer, got {keep_n_checkpoints}")

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_step = checkpoint_step
        self.keep_n_checkpoints = keep_n_checkpoints
        self._metadata = benchmark_metadata
        self.compression_level = compression_level

    def is_empty(self) -> bool:
        """Check if checkpoint directory is empty or doesn't exist."""
        if not self.checkpoint_dir.exists():
            return True
        return not any(self.checkpoint_dir.iterdir())

    def initialize(
        self,
        algorithms: list[Algorithm[Network]],
        problem: BenchmarkProblem,
        n_trials: int,
    ) -> None:
        """
        Initialize checkpoint directory structure for a new benchmark run.

        Args:
            algorithms: List of Algorithm objects to be benchmarked.
            problem: BenchmarkProblem configuration for the benchmark.
            n_trials: Total number of trials to run for each algorithm, used for resuming.

        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata: dict[str, Any] = {
            "n_trials": n_trials,
            "algorithms": [
                {
                    "name": alg.name,
                    "iterations": alg.iterations,
                    "index": idx,
                }
                for idx, alg in enumerate(algorithms)
            ],
        }
        if self._metadata is not None:
            metadata["benchmark_metadata"] = self._metadata
        if iop.get_seed() is not None:
            metadata["rng_seed"] = iop.get_seed()

        # Save initial state and metadata for resuming later if needed
        self._save_metadata(metadata)
        self._save_initial_algorithms(algorithms)
        self._save_benchmark_problem(problem)

        # Create algorithm directories
        for idx in range(len(algorithms)):
            self._get_algorithm_dir(idx).mkdir(parents=True, exist_ok=True)

        LOGGER.info(f"Initialized checkpoint directory at '{self.checkpoint_dir}'")

    def create_backup(self) -> Path:
        """
        Create a backup of the existing checkpoint directory.

        Returns:
            Path to the created backup zip file.

        Raises:
            FileExistsError: If the backup file already exists.

        """
        backup_path = Path(f"{self.checkpoint_dir}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")  # noqa: DTZ005
        if backup_path.exists():
            raise FileExistsError(f"Backup file '{backup_path}' already exists")

        shutil.make_archive(str(backup_path.with_suffix("")), "zip", self.checkpoint_dir)
        LOGGER.info(f"Created backup of checkpoint directory at '{backup_path}'")
        return backup_path

    def append_metadata(self, additional_metadata: dict[str, Any]) -> dict[str, Any]:
        """
        Append additional metadata to existing checkpoint metadata.

        This can be used to add information after initialization, such as system resource usage,
        hyperparameters, or other contextual information that may be relevant for analyzing results later.

        Args:
            additional_metadata: Dictionary of additional metadata to append to the existing metadata.

        Returns:
            Updated metadata dictionary after appending the additional metadata.

        """
        metadata = self.load_metadata()
        if "benchmark_metadata" not in metadata:
            metadata["benchmark_metadata"] = {}
        metadata["benchmark_metadata"].update(additional_metadata)
        self._save_metadata(metadata)
        return metadata

    def load_initial_algorithms(self) -> list[Algorithm[Network]]:
        """
        Load initial algorithm states from checkpoint.

        Returns:
            List of Algorithm objects representing the initial algorithm states.

        """
        initial_path = self._resolve_data_file("initial_algorithms.pkl.zst", "initial_algorithms.pkl")
        return cast("list[Algorithm[Network]]", self._load_pickle(initial_path))

    def load_benchmark_problem(self) -> BenchmarkProblem:
        """
        Load benchmark problem configuration from checkpoint.

        Returns:
            BenchmarkProblem object representing the benchmark problem configuration.

        """
        problem_path = self._resolve_data_file("benchmark_problem.pkl.zst", "benchmark_problem.pkl")
        return cast("BenchmarkProblem", self._load_pickle(problem_path))

    def should_checkpoint(self, iteration: int) -> bool:
        """
        Determine if a checkpoint should be saved at the current iteration.

        Checkpointing occurs if:
            - checkpoint_step is set and iteration is a multiple of checkpoint_step

        Args:
            iteration: Current iteration number.

        Returns:
            True if a checkpoint should be saved, False otherwise.

        Raises:
            ValueError: If iteration number is negative.

        """
        if self.checkpoint_step is None:
            return False

        if iteration < 0:
            raise ValueError(f"Iteration number must be non-negative, got {iteration}")

        return (iteration + 1) % self.checkpoint_step == 0

    def save_checkpoint(
        self,
        *,
        alg_idx: int,
        trial: int,
        iteration: int,
        algorithm: Algorithm[Network],
        network: Network,
        rng_state: dict[str, Any],
    ) -> Path:
        """
        Save checkpoint for a specific algorithm trial at a given iteration.

        Args:
            alg_idx: Algorithm index (0-based).
            trial: Trial number (0-based).
            iteration: Current iteration number.
            algorithm: Algorithm object with current internal state.
            network: Network object with current agent states and metrics.
            rng_state: RNG snapshot for deterministic resume.

        Returns:
            Path to the saved checkpoint file.

        """
        trial_dir = self._get_trial_dir(alg_idx, trial)
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Save both algorithm and network in a single pickle file to preserve shared object references
        checkpoint_path = trial_dir / f"checkpoint_{iteration:07d}.pkl.zst"
        progress_path = trial_dir / "progress.json"

        # Check if checkpoint already exists to avoid overwriting
        if checkpoint_path.exists() and progress_path.exists():
            with progress_path.open(encoding="utf-8") as f:
                progress = json.load(f)
            last_completed_iteration: int = progress.get("last_completed_iteration", -1)
            if last_completed_iteration == iteration:
                LOGGER.debug(
                    f"Checkpoint already exists for alg={alg_idx}, trial={trial}, iter={iteration}, skipping save"
                )
                return checkpoint_path

        checkpoint_data: _CheckpointData = {
            "algorithm": algorithm,
            "network": network,
            "iteration": iteration,
            "rng_state": rng_state,
        }
        self._save_pickle(checkpoint_path, checkpoint_data)

        # Update progress
        progress = {"last_completed_iteration": iteration}
        with progress_path.open("w", encoding="utf-8") as f:
            json.dump(progress, f)

        LOGGER.debug(f"Saved checkpoint: alg={alg_idx}, trial={trial}, iter={iteration}")

        self._cleanup_old_checkpoints(alg_idx, trial)
        return checkpoint_path

    def load_checkpoint(
        self, alg_idx: int, trial: int
    ) -> tuple[Algorithm[Network], Network, int, dict[str, Any]] | None:
        """
        Load the latest checkpoint for a specific algorithm trial.

        Args:
            alg_idx: Algorithm index (0-based).
            trial: Trial number (0-based).

        Returns:
            Tuple of (algorithm, network, last_iteration, rng_state) or None if no checkpoint exists.
            Execution should resume from iteration (last_iteration + 1).

        """
        trial_dir = self._get_trial_dir(alg_idx, trial)
        progress_path = trial_dir / "progress.json"

        if not progress_path.exists():
            return None

        # Load progress
        with progress_path.open(encoding="utf-8") as f:
            progress = json.load(f)
        last_iteration: int = progress["last_completed_iteration"]

        # Load both algorithm and network from single checkpoint file
        checkpoint_path = self._resolve_checkpoint_path(trial_dir, last_iteration)
        checkpoint_data = cast("_CheckpointData", self._load_pickle(checkpoint_path))

        algorithm: Algorithm[Network] = checkpoint_data["algorithm"]
        network: Network = checkpoint_data["network"]
        rng_state = checkpoint_data["rng_state"]

        LOGGER.debug(f"Loaded checkpoint: alg={alg_idx}, trial={trial}, iter={last_iteration}")
        return algorithm, network, last_iteration, rng_state

    def mark_trial_complete(
        self,
        *,
        alg_idx: int,
        trial: int,
        iteration: int,
        algorithm: Algorithm[Network],
        network: Network,
        rng_state: dict[str, Any],
    ) -> Path:
        """
        Mark a trial as complete and save final result.

        Args:
            alg_idx: Algorithm index (0-based).
            trial: Trial number (0-based).
            iteration: The final iteration number.
            algorithm: Final Algorithm state after all iterations complete.
            network: Final Network state after all iterations complete.
            rng_state: RNG snapshot for deterministic resume.

        Returns:
            Path to the saved final checkpoint file.

        """
        checkpoint_path = self.save_checkpoint(
            alg_idx=alg_idx,
            trial=trial,
            iteration=iteration,
            algorithm=algorithm,
            network=network,
            rng_state=rng_state,
        )

        # Mark as complete
        trial_dir = self._get_trial_dir(alg_idx, trial)
        complete_path = trial_dir / "complete.json"
        completed_metadata = {
            "alg_name": algorithm.name,
            "alg_idx": alg_idx,
            "trial": trial,
            "iteration": iteration,
            "checkpoint_path": str(checkpoint_path.name),
        }
        with complete_path.open("w") as f:
            json.dump(completed_metadata, f)

        LOGGER.debug(f"Marked trial complete: alg={alg_idx}, trial={trial}")
        return checkpoint_path

    def unmark_trial_complete(self, alg_idx: int, trial: int) -> None:
        """
        Remove the completion marker for a trial, allowing it to be rerun.

        Args:
            alg_idx: Algorithm index (0-based).
            trial: Trial number (0-based).

        """
        trial_dir = self._get_trial_dir(alg_idx, trial)
        complete_path = trial_dir / "complete.json"
        if complete_path.exists():
            complete_path.unlink()
            LOGGER.debug(f"Unmarked trial complete: alg={alg_idx}, trial={trial}")

    def is_trial_complete(self, alg_idx: int, trial: int) -> bool:
        """
        Check if a trial has been completed.

        Args:
            alg_idx: Algorithm index (0-based).
            trial: Trial number (0-based).

        Returns:
            True if the trial has completed, False otherwise.

        """
        trial_dir = self._get_trial_dir(alg_idx, trial)
        return (trial_dir / "complete.json").exists()

    def is_benchmark_started(self) -> bool:
        """
        Check if the benchmark has been started by looking for any existing checkpoints.

        Returns:
            True if any trial has at least one checkpoint saved, False otherwise.

        """
        metadata = self.load_metadata()
        if metadata is None or "n_trials" not in metadata or "algorithms" not in metadata:
            return False

        n_trials = metadata["n_trials"]
        algorithms = metadata["algorithms"]
        for alg in algorithms:
            alg_idx = alg["index"]
            for trial in range(n_trials):
                trial_dir = self._get_trial_dir(alg_idx, trial)
                if any(trial_dir.glob("checkpoint_*.pkl*")):
                    return True
        return False

    def is_benchmark_completed(self) -> bool:
        """
        Check if all trials for all algorithms have been completed.

        Returns:
            True if all trials for all algorithms are marked as complete, False otherwise.

        """
        metadata = self.load_metadata()

        if metadata is None or "n_trials" not in metadata or "algorithms" not in metadata:
            return False

        n_trials = metadata["n_trials"]
        algorithms = metadata["algorithms"]
        for alg in algorithms:
            alg_idx = alg["index"]
            for trial in range(n_trials):
                if not self.is_trial_complete(alg_idx, trial):
                    return False
        return True

    def are_metrics_computed(self) -> bool:
        """
        Check if the metrics have been computed and saved in the checkpoint.

        Returns:
            True if the metrics result file exists, False otherwise.

        """
        metric_path = self.checkpoint_dir / "metric_computation_complete.json"
        return metric_path.exists()

    def load_trial_result(self, alg_idx: int, trial: int) -> tuple[Algorithm[Network], Network]:
        """
        Load final result of a completed trial.

        Args:
            alg_idx: Algorithm index (0-based).
            trial: Trial number (0-based).

        Returns:
            Tuple of (Algorithm object, Network object) with final state after all iterations.

        Raises:
            ValueError: If the trial is not marked as complete or if the checkpoint data is invalid.

        """
        trial_dir = self._get_trial_dir(alg_idx, trial)
        complete_path = trial_dir / "complete.json"

        if not complete_path.exists():
            raise ValueError(f"Trial {trial} for algorithm index {alg_idx} is not marked as complete")

        with complete_path.open(encoding="utf-8") as f:
            completed_metadata = json.load(f)
        checkpoint_path = Path(completed_metadata["checkpoint_path"])
        final_path = trial_dir / checkpoint_path.name

        checkpoint_data = cast("_CheckpointData", self._load_pickle(final_path))

        alg: Algorithm[Network] = checkpoint_data["algorithm"]
        network: Network = checkpoint_data["network"]
        return alg, network

    def get_completed_trials(self, alg_idx: int, n_trials: int) -> list[int]:
        """
        Get list of completed trial numbers for an algorithm.

        Args:
            alg_idx: Algorithm index (0-based).
            n_trials: Total number of trials in the benchmark.

        Returns:
            List of completed trial numbers (0-based).

        """
        return [trial for trial in range(n_trials) if self.is_trial_complete(alg_idx, trial)]

    def load_metadata(self) -> dict[str, Any]:
        """
        Load checkpoint metadata.

        If no metadata file exists, returns an empty dictionary.

        Returns:
            Dictionary containing benchmark_metadata and algorithms list.

        """
        metadata_path = self.checkpoint_dir / "metadata.json"

        if not metadata_path.exists():
            return {}

        with metadata_path.open(encoding="utf-8") as f:
            metadata: dict[str, Any] = json.load(f)
        return metadata

    def load_benchmark_result(self) -> BenchmarkResult:
        """
        Load benchmark problem configuration and states from checkpoint.

        If an algorithm does not have all trials completed, its results will be skipped and not included in the loaded
        benchmark result. This is to ensure that the metrics are not skewed by incomplete data and only include
        algorithms with full results. A warning will be logged for any incomplete algorithms.

        Returns:
            BenchmarkResult object containing the loaded benchmark problem, initial algorithms, and initial network.

        """
        progress_bar_threshold = 1_000  # How many MB of checkpoint data should trigger showing a progress bar
        progress_bar = self.checkpoint_size() > progress_bar_threshold
        problem = self.load_benchmark_problem()
        algorithms = self.load_initial_algorithms()
        metadata = self.load_metadata()
        n_trials = metadata["n_trials"]
        states: dict[Algorithm[Network], list[Network]] = {}
        for idx, alg in track(
            enumerate(algorithms),
            total=len(algorithms),
            description="Loading benchmark results...",
            transient=True,
            disable=not progress_bar,
        ):
            completed_trials = self.get_completed_trials(idx, n_trials)
            if len(completed_trials) != n_trials:
                LOGGER.warning(
                    f"Algorithm '{alg.name}' has {len(completed_trials)}/{n_trials} completed trials. "
                    "Results will not be loaded for this algorithm."
                )
                continue
            for trial in completed_trials:
                loaded_alg, loaded_net = self.load_trial_result(idx, trial)
                if loaded_alg.name != alg.name:
                    LOGGER.warning(
                        f"Algorithm mismatch in trial {trial} for algorithm {alg.name}, loaded {loaded_alg.name}. "
                        "Results will not be loaded for this algorithm."
                    )
                    states.pop(alg, None)  # Remove any previously loaded states for this algorithm
                    break
                if alg not in states:
                    states[alg] = []
                states[alg].append(loaded_net)

        return BenchmarkResult(
            problem=problem,
            states=states,
        )

    def save_metrics_result(self, metrics_result: MetricResult) -> None:
        """
        Save the computed metrics result to the checkpoint directory.

        Args:
            metrics_result: MetricsResult object containing the computed metrics to save.

        """
        metric_path = self.checkpoint_dir / "metric_computation.pkl.zst"
        metric_marker_path = self.checkpoint_dir / "metric_computation_complete.json"

        # Remove agent metrics from checkpoint to save space (can be a lot),
        # this can be loaded again from the benchmark result
        if self.is_benchmark_completed():
            metrics_result.agent_metrics = None

        self._save_pickle(metric_path, metrics_result)

        # Save a small marker file to indicate that metric computation was saved successfully.
        # This is used to avoid issues where the process is killed while writing the potentially
        # large metric_computation.pkl.zst file,
        with metric_marker_path.open("w") as f:
            json.dump({"metric_computation_complete": True}, f)

        LOGGER.info(f"Saved computed metrics result to {metric_path}")

    def load_metrics_result(self, skip_agent_metrics: bool = False) -> MetricResult:
        """
        Load the computed metrics result from the checkpoint directory.

        Args:
            skip_agent_metrics: If True, do not attempt to load agent metrics from the benchmark
                result if they are not present in the checkpoint. This can save time if agent metrics
                are not needed for the intended analysis, which can be useful for automatic analysis.
                Agent metrics are needed for :class:`~decent_bench.metrics.ComputationalCost` and may be used if
                :class:`~decent_bench.costs.EmpiricalRiskCost` is used.

        Returns:
            MetricsResult object containing the computed metrics.

        """
        metric_path = self._resolve_data_file("metric_computation.pkl.zst", "metric_computation.pkl")
        metrics_result = cast("MetricResult", self._load_pickle(metric_path))

        if metrics_result.agent_metrics is None and not skip_agent_metrics:
            try:
                benchmark_result = self.load_benchmark_result()
                resulting_agent_states: dict[Algorithm[Network], list[list[AgentMetricsView]]] = {}
                for alg, networks in benchmark_result.states.items():
                    algorithms = list(metrics_result.table_results or metrics_result.plot_results or [])
                    original_alg = next((a for a in algorithms if a.name == alg.name), None)
                    if original_alg is None:
                        LOGGER.warning(
                            f"Original algorithm '{alg.name}' not found in benchmark problem configuration. "
                            "Cannot reconstruct agent metrics for this algorithm."
                        )
                        continue
                    resulting_agent_states[original_alg] = [
                        [AgentMetricsView.from_agent(a) for a in nw.agents()] for nw in networks
                    ]
                metrics_result.agent_metrics = resulting_agent_states
            except Exception as e:
                LOGGER.warning(
                    f"Failed to load benchmark result to reconstruct agent metrics: {e}"
                    "Some functionality may be limited without agent metrics available."
                )
                metrics_result.agent_metrics = None

        LOGGER.info(f"Loaded computed metrics result from {metric_path}")
        return metrics_result

    def get_results_path(self) -> Path:
        """
        Get the path to the results directory within the checkpoint directory.

        Returns:
            Path to the results directory within the checkpoint directory.

        """
        return self.checkpoint_dir / "results"

    def clear(self) -> None:
        """
        Remove entire checkpoint directory and all its contents.

        Warning:
            This permanently deletes all checkpoint data.

        """
        if self.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_dir)
            LOGGER.info(f"Cleared checkpoint directory: {self.checkpoint_dir}")

    def checkpoint_size(self) -> int:
        """
        Calculate the total size of all checkpoint files in MB.

        Returns:
            Total size of checkpoint files in MB.

        """
        total_size = 0
        for file in self.checkpoint_dir.rglob("*"):
            if file.is_file():
                total_size += file.stat().st_size
        return total_size // (1024 * 1024)  # Convert to MB

    def _get_algorithm_dir(self, alg_idx: int) -> Path:
        """Get directory path for an algorithm."""
        return self.checkpoint_dir / f"algorithm_{alg_idx}"

    def _get_trial_dir(self, alg_idx: int, trial: int) -> Path:
        """Get directory path for a specific trial."""
        return self._get_algorithm_dir(alg_idx) / f"trial_{trial}"

    def _save_metadata(self, metadata: dict[str, Any]) -> None:
        """Save metadata to checkpoint directory."""
        metadata_path = self.checkpoint_dir / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f)
        LOGGER.debug(f"Saved metadata to {metadata_path}")

    def _save_initial_algorithms(self, algorithms: list[Algorithm[Network]]) -> None:
        """Save initial algorithm states before any trials run."""
        initial_path = self.checkpoint_dir / "initial_algorithms.pkl.zst"
        with Status(f"Saving initial algorithms to {initial_path}..."):
            self._save_pickle(initial_path, algorithms)
        LOGGER.debug(f"Saved initial algorithms to {initial_path}")

    def _save_benchmark_problem(self, problem: BenchmarkProblem) -> None:
        """Save benchmark problem configuration."""
        problem_path = self.checkpoint_dir / "benchmark_problem.pkl.zst"
        with Status(f"Saving benchmark problem configuration to {problem_path}..."):
            self._save_pickle(problem_path, problem)
        LOGGER.debug(f"Saved benchmark problem to {problem_path}")

    def _resolve_data_file(self, preferred_name: str, legacy_name: str) -> Path:
        """Resolve a data file path, preferring the current format with legacy fallback."""
        preferred = self.checkpoint_dir / preferred_name
        if preferred.exists():
            return preferred

        legacy = self.checkpoint_dir / legacy_name
        if legacy.exists():
            return legacy

        return preferred

    def _resolve_checkpoint_path(self, trial_dir: Path, iteration: int) -> Path:
        """Resolve a checkpoint path for an iteration with current/legacy extension support."""
        preferred = trial_dir / f"checkpoint_{iteration:07d}.pkl.zst"
        if preferred.exists():
            return preferred

        legacy = trial_dir / f"checkpoint_{iteration:07d}.pkl"
        if legacy.exists():
            return legacy

        return preferred

    def _checkpoint_iteration(self, path: Path) -> int:
        """
        Extract checkpoint iteration from a checkpoint filename.

        Raises:
            ValueError: If the filename is not a valid checkpoint name.

        """
        match = _CHECKPOINT_NAME_RE.match(path.name)
        if match is None:
            raise ValueError(f"Invalid checkpoint filename: {path.name}")
        return int(match.group(1))

    def _save_pickle(self, path: Path, data: object) -> None:
        """Save Python object as zstd-compressed pickle payload."""
        compressor = zstd.ZstdCompressor(level=self.compression_level)
        with path.open("wb") as file_obj, compressor.stream_writer(file_obj) as compressed_writer:
            pickle.dump(data, compressed_writer, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_pickle(self, path: Path) -> object:
        """Load pickle payload, supporting both zstd-compressed and legacy uncompressed files."""
        with path.open("rb") as file_obj:
            magic = file_obj.read(len(_ZSTD_MAGIC))
            file_obj.seek(0)
            if magic == _ZSTD_MAGIC:
                decompressor = zstd.ZstdDecompressor()
                with decompressor.stream_reader(file_obj) as decompressed_reader:
                    return pickle.load(decompressed_reader)  # noqa: S301
            # Fall back to legacy uncompressed pickle for backward compatibility
            return pickle.load(file_obj)  # noqa: S301

    def _cleanup_old_checkpoints(self, alg_idx: int, trial: int) -> None:
        """
        Remove old iteration checkpoint files, keeping only the most recent N.

        Args:
            alg_idx: Algorithm index (0-based).
            trial: Trial number (0-based).

        """
        trial_dir = self._get_trial_dir(alg_idx, trial)
        if not trial_dir.exists():
            return

        # Find all iteration checkpoint files
        checkpoint_files = [
            *trial_dir.glob("checkpoint_*.pkl"),
            *trial_dir.glob("checkpoint_*.pkl.zst"),
        ]
        # Sort by iteration number in filename (checkpoint_0000100.pkl.zst -> 100)
        checkpoint_files.sort(key=self._checkpoint_iteration, reverse=True)

        # Remove older checkpoints
        if len(checkpoint_files) > self.keep_n_checkpoints:
            for file_to_remove in checkpoint_files[self.keep_n_checkpoints :]:
                try:
                    file_to_remove.unlink()
                    LOGGER.debug(f"Removed old checkpoint: {file_to_remove}")
                except FileNotFoundError:
                    LOGGER.debug(f"Checkpoint file already removed by another process: {file_to_remove}")
