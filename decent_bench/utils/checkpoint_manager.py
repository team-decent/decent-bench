import json
import pickle  # noqa: S403
import shutil
from pathlib import Path
from typing import Any

from decent_bench.distributed_algorithms import Algorithm
from decent_bench.networks import P2PNetwork
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
            ├── metadata.json                 # Run configuration and algorithm metadata
            ├── initial_network.pkl           # Initial network state (before any trials)
            └── algorithm_0/                  # Directory for first algorithm
                ├── trial_0/                  # Directory for trial 0
                │   ├── iter_0000100.pkl      # Network state at iteration 100
                │   ├── iter_0000200.pkl      # Network state at iteration 200
                │   ├── algorithm_state.pkl   # Algorithm object state (updated each checkpoint)
                │   ├── progress.json         # {"last_completed_iteration": N}
                │   └── complete.json         # Marker file, contains path to final checkpoint
                ├── trial_1/
                │   └── ...
                └── trial_N/
                    └── ...

    File Descriptions:
        - **metadata.json**: Benchmark configuration.
        - **initial_network.pkl**: Starting network state before any algorithm execution.
        - **iter_NNNNNNN.pkl**: Network state snapshots at specific iterations during execution.
        - **algorithm_state.pkl**: Algorithm object with internal state at last checkpoint.
        - **progress.json**: Tracks the last completed iteration within a trial.
        - **complete.json**: Marker file, contains path to final checkpoint.

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

    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        checkpoint_step: int | None,
        keep_n_checkpoints: int,
    ) -> None:
        """Initialize CheckpointManager with a checkpoint directory path."""
        # Checked
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_step = checkpoint_step
        self.keep_n_checkpoints = keep_n_checkpoints

    def is_empty(self) -> bool:
        """Check if checkpoint directory is empty or doesn't exist."""
        # Checked
        if not self.checkpoint_dir.exists():
            return True
        return not any(self.checkpoint_dir.iterdir())

    def initialize(self, algorithms: list[Algorithm], network: P2PNetwork, benchmark_metadata: dict[str, Any]) -> None:
        """
        Initialize checkpoint directory structure for a new benchmark run.

        Args:
            algorithms: List of Algorithm objects to be benchmarked.
            network: Initial P2PNetwork state before any trials run.
            benchmark_metadata: Benchmark configuration (n_trials, checkpoint_step).

        """
        # Checked
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            "benchmark_metadata": benchmark_metadata,
            "algorithms": [
                {
                    "name": alg.name,
                    "iterations": alg.iterations,
                    "index": idx,
                }
                for idx, alg in enumerate(algorithms)
            ],
        }
        self._save_metadata(metadata)
        self._save_initial_network(network)

        # Create algorithm directories
        for idx in range(len(algorithms)):
            self._get_algorithm_dir(idx).mkdir(parents=True, exist_ok=True)

    def load_initial_network(self) -> P2PNetwork:
        """
        Load initial network state from checkpoint.

        Returns:
            P2PNetwork object representing the initial network state.

        """
        initial_path = self.checkpoint_dir / "initial_network.pkl"
        with initial_path.open("rb") as f:
            ret: P2PNetwork = pickle.load(f)  # noqa: S301
        return ret

    def should_checkpoint(self, alg_iterations: int, iteration: int) -> bool:
        """
        Determine if a checkpoint should be saved at the current iteration.

        Checkpointing occurs if:
            - checkpoint_step is set and iteration is a multiple of checkpoint_step
            - This is the final iteration of the algorithm.

        Args:
            alg_iterations: Total number of iterations for the algorithm.
            iteration: Current iteration number.

        Returns:
            True if a checkpoint should be saved, False otherwise.

        """
        if iteration >= alg_iterations - 1:
            return True  # Always checkpoint at the final iteration

        if self.checkpoint_step is None:
            return False

        return iteration % self.checkpoint_step == 0

    def save_checkpoint(
        self,
        alg_idx: int,
        trial: int,
        iteration: int,
        algorithm: Algorithm,
        network: P2PNetwork,
    ) -> tuple[Path, Path]:
        """
        Save checkpoint for a specific algorithm trial at a given iteration.

        Saves three files: network state, algorithm state, and progress tracking.

        Args:
            alg_idx: Algorithm index (0-based).
            trial: Trial number (0-based).
            iteration: Current iteration number.
            algorithm: Algorithm object with current internal state.
            network: P2PNetwork object with current agent states and metrics.

        Returns:
            Tuple of (network_checkpoint_path, algorithm_checkpoint_path) for the saved checkpoint files.

        """
        trial_dir = self._get_trial_dir(alg_idx, trial)
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Save network state
        network_path = trial_dir / f"network_{iteration:07d}.pkl"
        with network_path.open("wb") as f:
            pickle.dump(network, f)

        # Save algorithm state
        alg_path = trial_dir / f"algorithm_state_{iteration:07d}.pkl"
        with alg_path.open("wb") as f:
            pickle.dump(algorithm, f)

        # Update progress
        progress = {"last_completed_iteration": iteration}
        progress_path = trial_dir / "progress.json"
        with progress_path.open("w", encoding="utf-8") as f:
            json.dump(progress, f)

        LOGGER.debug(f"Saved checkpoint: alg={alg_idx}, trial={trial}, iter={iteration}")

        self._cleanup_old_checkpoints(alg_idx, trial)
        return network_path, alg_path

    def load_checkpoint(self, alg_idx: int, trial: int) -> tuple[Algorithm, P2PNetwork, int] | None:
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
        last_iteration = progress["last_completed_iteration"]

        # Load algorithm state
        alg_path = trial_dir / f"algorithm_state_{last_iteration:07d}.pkl"
        with alg_path.open("rb") as f:
            algorithm = pickle.load(f)  # noqa: S301

        # Load network state
        network_path = trial_dir / f"network_{last_iteration:07d}.pkl"
        with network_path.open("rb") as f:
            network = pickle.load(f)  # noqa: S301

        LOGGER.debug(f"Loaded checkpoint: alg={alg_idx}, trial={trial}, iter={last_iteration}")
        return algorithm, network, last_iteration

    def mark_trial_complete(
        self,
        alg_idx: int,
        trial: int,
        iteration: int,
        algorithm: Algorithm,
        network: P2PNetwork,
    ) -> tuple[Path, Path]:
        """
        Mark a trial as complete and save final result.

        Args:
            alg_idx: Algorithm index (0-based).
            trial: Trial number (0-based).
            iteration: The final iteration number.
            algorithm: Final Algorithm state after all iterations complete.
            network: Final P2PNetwork state after all iterations complete.

        Returns:
            Tuple of (network_checkpoint_path, algorithm_checkpoint_path) for the saved final checkpoint files.

        """
        network_path, alg_path = self.save_checkpoint(alg_idx, trial, iteration, algorithm, network)

        # Mark as complete
        trial_dir = self._get_trial_dir(alg_idx, trial)
        complete_path = trial_dir / "complete.json"
        completed_metadata = {
            "alg_name": algorithm.name,
            "alg_idx": alg_idx,
            "trial": trial,
            "iteration": iteration,
            "network_path": str(network_path),
            "algorithm_path": str(alg_path),
        }
        with complete_path.open("w") as f:
            json.dump(completed_metadata, f)

        LOGGER.debug(f"Marked trial complete: alg={alg_idx}, trial={trial}")
        return network_path, alg_path

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

    def load_trial_result(self, alg_idx: int, trial: int) -> P2PNetwork:
        """
        Load final result of a completed trial.

        Args:
            alg_idx: Algorithm index (0-based).
            trial: Trial number (0-based).

        Returns:
            P2PNetwork object with final state after all iterations.

        """
        trial_dir = self._get_trial_dir(alg_idx, trial)
        complete_path = trial_dir / "complete.json"

        with complete_path.open(encoding="utf-8") as f:
            completed_metadata = json.load(f)
        final_path = Path(completed_metadata["network_path"])

        with final_path.open("rb") as f:
            ret: P2PNetwork = pickle.load(f)  # noqa: S301
        return ret

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

    def _save_initial_network(self, network: P2PNetwork) -> None:
        """Save initial network state before any trials run."""
        initial_path = self.checkpoint_dir / "initial_network.pkl"
        with initial_path.open("wb") as f:
            pickle.dump(network, f)
        LOGGER.debug(f"Saved initial network to {initial_path}")

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
        checkpoint_files = sorted(trial_dir.glob("network_*.pkl")) + sorted(trial_dir.glob("algorithm_state_*.pkl"))
        checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        # Remove older checkpoints
        if len(checkpoint_files) > self.keep_n_checkpoints * 2:
            for file_to_remove in checkpoint_files[self.keep_n_checkpoints * 2 :]:
                file_to_remove.unlink()
                LOGGER.debug(f"Removed old checkpoint: {file_to_remove}")
