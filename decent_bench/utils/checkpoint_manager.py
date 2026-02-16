import json
import pickle  # noqa: S403
import shutil
from pathlib import Path
from typing import Any

from decent_bench.benchmark_problem import BenchmarkProblem
from decent_bench.distributed_algorithms import Algorithm
from decent_bench.networks import Network
from decent_bench.utils.logger import LOGGER


class CheckpointManager:
    """
    Manages checkpoint directory structure and file operations for benchmark execution.

    The CheckpointManager creates and maintains a hierarchical directory structure for storing
    checkpoint data during benchmark execution. This allows benchmarks to be resumed if interrupted,
    and provides incremental saving of results as trials complete.

    Directory Structure:
        The checkpoint directory is organized as follows::

            checkpoint_dir/
            ├── metadata.json                   # Run configuration and algorithm metadata
            ├── benchmark_problem.pkl           # Initial benchmark problem state (before any trials)
            ├── initial_algorithms.pkl          # Initial algorithm states (before any trials)
            ├── initial_network.pkl             # Initial network state (before any trials)
            ├── algorithm_0/                    # Directory for first algorithm
            │   ├── trial_0/                    # Directory for trial 0
            │   │   ├── checkpoint_0000100.pkl  # Combined algorithm+network state at iteration 100
            │   │   ├── checkpoint_0000200.pkl  # Combined algorithm+network state at iteration 200
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
        - **benchmark_problem.pkl**: Initial benchmark problem state before any trials run.
        - **initial_algorithms.pkl**: Initial algorithm states before any trials run.
        - **initial_network.pkl**: Starting network state before any algorithm execution.
        - **checkpoint_NNNNNNN.pkl**: Combined checkpoint containing both algorithm and network state.
          This preserves shared object references and ensures consistency between algorithm and network
          states at each checkpoint. The checkpoint data is a dictionary with the following structure:

            - algorithm: :class:`~decent_bench.distributed_algorithms.Algorithm`
            - network: :class:`~decent_bench.networks.Network`
            - iteration: iteration

          where "algorithm" is the :class:`~decent_bench.distributed_algorithms.Algorithm` object with its internal
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
        checkpoint_dir: Path to the checkpoint directory.
        checkpoint_step: Number of iterations between checkpoints within each trial.
            If None, only save at trial completion.
        keep_n_checkpoints: Maximum number of iteration checkpoints to keep per trial.
            Older checkpoints are automatically deleted to save disk space.

    Raises:
            ValueError: If checkpoint_step is not a positive integer or None.

    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        checkpoint_step: int | None,
        keep_n_checkpoints: int,
    ) -> None:
        """
        Initialize CheckpointManager with a checkpoint directory path.

        Args:
        checkpoint_dir: Path to the checkpoint directory.
        checkpoint_step: Number of iterations between checkpoints within each trial.
            If None, only save at trial completion.
        keep_n_checkpoints: Maximum number of iteration checkpoints to keep per trial.
            Older checkpoints are automatically deleted to save disk space.

        Raises:
            ValueError: If checkpoint_step is not a positive integer or None.
            ValueError: If keep_n_checkpoints is not a positive integer.

        """
        if checkpoint_step is not None and checkpoint_step <= 0:
            raise ValueError(f"checkpoint_step must be a positive integer or None, got {checkpoint_step}")
        if keep_n_checkpoints <= 0:
            raise ValueError(f"keep_n_checkpoints must be a positive integer, got {keep_n_checkpoints}")

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_step = checkpoint_step
        self.keep_n_checkpoints = keep_n_checkpoints

    def is_empty(self) -> bool:
        """Check if checkpoint directory is empty or doesn't exist."""
        if not self.checkpoint_dir.exists():
            return True
        return not any(self.checkpoint_dir.iterdir())

    def initialize(
        self,
        algorithms: list[Algorithm],
        network: Network,
        problem: BenchmarkProblem,
        n_trials: int,
        benchmark_metadata: dict[str, Any] | None,
    ) -> None:
        """
        Initialize checkpoint directory structure for a new benchmark run.

        Args:
            algorithms: List of Algorithm objects to be benchmarked.
            network: Initial Network state before any trials run.
            problem: BenchmarkProblem configuration for the benchmark.
            n_trials: Total number of trials to run for each algorithm, used for resuming.
            benchmark_metadata: Optional dictionary of additional metadata to save in the checkpoint directory,
                such as hyperparameters or system information. This can be useful for keeping track of the benchmark
                configuration and context when analyzing results later.

        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
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
        if benchmark_metadata is not None:
            metadata["benchmark_metadata"] = benchmark_metadata

        # Save initial state and metadata for resuming later if needed
        self._save_metadata(metadata)
        self._save_initial_network(network)
        self._save_initial_algorithms(algorithms)
        self._save_benchmark_problem(problem)

        # Create algorithm directories
        for idx in range(len(algorithms)):
            self._get_algorithm_dir(idx).mkdir(parents=True, exist_ok=True)

    def create_backup(self) -> Path:
        """
        Create a backup of the existing checkpoint directory.

        Returns:
            Path to the created backup zip file.

        Raises:
            FileExistsError: If the backup file already exists.

        """
        backup_path = Path(f"{self.checkpoint_dir}_backup.zip")
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

    def load_initial_network(self) -> Network:
        """
        Load initial network state from checkpoint.

        Returns:
            Network object representing the initial network state.

        """
        initial_path = self.checkpoint_dir / "initial_network.pkl"
        with initial_path.open("rb") as f:
            ret: Network = pickle.load(f)  # noqa: S301
        return ret

    def load_initial_algorithms(self) -> list[Algorithm]:
        """
        Load initial algorithm states from checkpoint.

        Returns:
            List of Algorithm objects representing the initial algorithm states.

        """
        initial_path = self.checkpoint_dir / "initial_algorithms.pkl"
        with initial_path.open("rb") as f:
            ret: list[Algorithm] = pickle.load(f)  # noqa: S301
        return ret

    def load_benchmark_problem(self) -> BenchmarkProblem:
        """
        Load benchmark problem configuration from checkpoint.

        Returns:
            BenchmarkProblem object representing the benchmark problem configuration.

        """
        problem_path = self.checkpoint_dir / "benchmark_problem.pkl"
        with problem_path.open("rb") as f:
            ret: BenchmarkProblem = pickle.load(f)  # noqa: S301
        return ret

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
        alg_idx: int,
        trial: int,
        iteration: int,
        algorithm: Algorithm,
        network: Network,
    ) -> Path:
        """
        Save checkpoint for a specific algorithm trial at a given iteration.

        Args:
            alg_idx: Algorithm index (0-based).
            trial: Trial number (0-based).
            iteration: Current iteration number.
            algorithm: Algorithm object with current internal state.
            network: Network object with current agent states and metrics.

        Returns:
            Path to the saved checkpoint file.

        """
        trial_dir = self._get_trial_dir(alg_idx, trial)
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Save both algorithm and network in a single pickle file to preserve shared object references
        checkpoint_path = trial_dir / f"checkpoint_{iteration:07d}.pkl"
        checkpoint_data = {
            "algorithm": algorithm,
            "network": network,
            "iteration": iteration,
        }
        with checkpoint_path.open("wb") as f:
            pickle.dump(checkpoint_data, f)

        # Update progress
        progress = {"last_completed_iteration": iteration}
        progress_path = trial_dir / "progress.json"
        with progress_path.open("w", encoding="utf-8") as f:
            json.dump(progress, f)

        LOGGER.debug(f"Saved checkpoint: alg={alg_idx}, trial={trial}, iter={iteration}")

        self._cleanup_old_checkpoints(alg_idx, trial)
        return checkpoint_path

    def load_checkpoint(self, alg_idx: int, trial: int) -> tuple[Algorithm, Network, int] | None:
        """
        Load the latest checkpoint for a specific algorithm trial.

        Args:
            alg_idx: Algorithm index (0-based).
            trial: Trial number (0-based).

        Returns:
            Tuple of (algorithm, network, last_iteration) or None if no checkpoint exists.
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
        checkpoint_path = trial_dir / f"checkpoint_{last_iteration:07d}.pkl"
        with checkpoint_path.open("rb") as f:
            checkpoint_data = pickle.load(f)  # noqa: S301

        algorithm: Algorithm = checkpoint_data["algorithm"]
        network: Network = checkpoint_data["network"]

        LOGGER.debug(f"Loaded checkpoint: alg={alg_idx}, trial={trial}, iter={last_iteration}")
        return algorithm, network, last_iteration

    def mark_trial_complete(
        self,
        alg_idx: int,
        trial: int,
        iteration: int,
        algorithm: Algorithm,
        network: Network,
    ) -> Path:
        """
        Mark a trial as complete and save final result.

        Args:
            alg_idx: Algorithm index (0-based).
            trial: Trial number (0-based).
            iteration: The final iteration number.
            algorithm: Final Algorithm state after all iterations complete.
            network: Final Network state after all iterations complete.

        Returns:
            Path to the saved final checkpoint file.

        """
        checkpoint_path = self.save_checkpoint(alg_idx, trial, iteration, algorithm, network)

        # Mark as complete
        trial_dir = self._get_trial_dir(alg_idx, trial)
        complete_path = trial_dir / "complete.json"
        completed_metadata = {
            "alg_name": algorithm.name,
            "alg_idx": alg_idx,
            "trial": trial,
            "iteration": iteration,
            "checkpoint_path": str(checkpoint_path),
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

    def load_trial_result(self, alg_idx: int, trial: int) -> tuple[Algorithm, Network]:
        """
        Load final result of a completed trial.

        Args:
            alg_idx: Algorithm index (0-based).
            trial: Trial number (0-based).

        Returns:
            Tuple of (Algorithm object, Network object) with final state after all iterations.

        """
        trial_dir = self._get_trial_dir(alg_idx, trial)
        complete_path = trial_dir / "complete.json"

        with complete_path.open(encoding="utf-8") as f:
            completed_metadata = json.load(f)
        final_path = Path(completed_metadata["checkpoint_path"])

        with final_path.open("rb") as f:
            checkpoint_data = pickle.load(f)  # noqa: S301

        alg: Algorithm = checkpoint_data["algorithm"]
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

        Returns:
            Dictionary containing benchmark_metadata and algorithms list.

        """
        metadata_path = self.checkpoint_dir / "metadata.json"
        with metadata_path.open(encoding="utf-8") as f:
            metadata: dict[str, Any] = json.load(f)
        return metadata

    def get_results_path(self, file_name: str) -> Path:
        """
        Get the path for a results file (e.g., table or plot) in the checkpoint directory.

        Args:
            file_name: Name of the results file (e.g., "table_results.txt" or "plots.png").

        Returns:
            Path object representing the full path to the results file within the checkpoint directory.

        """
        return self.checkpoint_dir / "results" / file_name

    def clear(self) -> None:
        """
        Remove entire checkpoint directory and all its contents.

        Warning:
            This permanently deletes all checkpoint data.

        """
        if self.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_dir)
            LOGGER.info(f"Cleared checkpoint directory: {self.checkpoint_dir}")

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

    def _save_initial_network(self, network: Network) -> None:
        """Save initial network state before any trials run."""
        initial_path = self.checkpoint_dir / "initial_network.pkl"
        with initial_path.open("wb") as f:
            pickle.dump(network, f)
        LOGGER.debug(f"Saved initial network to {initial_path}")

    def _save_initial_algorithms(self, algorithms: list[Algorithm]) -> None:
        """Save initial algorithm states before any trials run."""
        initial_path = self.checkpoint_dir / "initial_algorithms.pkl"
        with initial_path.open("wb") as f:
            pickle.dump(algorithms, f)
        LOGGER.debug(f"Saved initial algorithms to {initial_path}")

    def _save_benchmark_problem(self, problem: BenchmarkProblem) -> None:
        """Save benchmark problem configuration."""
        problem_path = self.checkpoint_dir / "benchmark_problem.pkl"
        with problem_path.open("wb") as f:
            pickle.dump(problem, f)
        LOGGER.debug(f"Saved benchmark problem to {problem_path}")

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
        checkpoint_files = list(trial_dir.glob("checkpoint_*.pkl"))
        # Sort by iteration number in filename (checkpoint_0000100.pkl -> 100)
        checkpoint_files.sort(key=lambda p: int(p.stem.split("_")[-1]), reverse=True)

        # Remove older checkpoints
        if len(checkpoint_files) > self.keep_n_checkpoints:
            for file_to_remove in checkpoint_files[self.keep_n_checkpoints :]:
                try:
                    file_to_remove.unlink()
                    LOGGER.debug(f"Removed old checkpoint: {file_to_remove}")
                except FileNotFoundError:
                    LOGGER.debug(f"Checkpoint file already removed by another process: {file_to_remove}")
