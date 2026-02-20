import contextlib
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from decent_bench.benchmark_problem import BenchmarkProblem

if TYPE_CHECKING:
    import queue

    from decent_bench.agents import Agent


class RuntimeMetric(ABC):
    """
    Abstract base class for runtime metrics.

    Runtime metrics are computed during algorithm execution to provide live feedback
    for early stopping or monitoring. Unlike post-hoc metrics, they don't store historical
    data and are designed to be lightweight.

    To create a new runtime metric, subclass this class and implement :meth:`description` and
    :meth:`compute`.

    Args:
        update_interval: Number of iterations between metric updates, do not update more frequently than necessary as
            this can slow down the algorithm.
        save_path: Path to save the plot when the metric is updated, if None, the plot will not be saved

    """

    def __init__(self, update_interval: int, save_path: str | Path | None = None) -> None:
        """
        Initialize runtime metric.

        Args:
            update_interval: Number of iterations between metric updates, do not update more frequently than necessary
                as this can slow down the algorithm.
            save_path: Path to save the plot when the metric is updated, if None, the plot will not be saved

        """
        self._update_interval = update_interval
        self._save_path = Path(save_path) if save_path is not None else None
        self._queue: queue.Queue[Any] | None = None
        self._metric_id: str = ""
        self._algorithm_name: str = ""
        self._trial: int = 0

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of the metric, used as the y-axis label."""

    @property
    def update_interval(self) -> int:
        """
        Number of iterations between metric updates.

        Returns:
            Number of iterations between updates.

        """
        return self._update_interval

    @abstractmethod
    def compute(self, problem: "BenchmarkProblem", agents: Sequence["Agent"], iteration: int) -> float:
        """
        Compute the metric value for the current iteration.

        Args:
            problem: benchmark problem being solved
            agents: sequence of agents with their current state
            iteration: current iteration number

        Returns:
            The computed metric value as a float.

        Note:
            This method should be efficient as it's called during algorithm execution.
            Avoid expensive computations or operations that might significantly slow down
            the algorithm.

        """

    def initialize_plot(self, algorithm_name: str, trial: int, queue: "queue.Queue[Any]") -> None:
        """
        Initialize the plot for this metric.

        Sends initialization message to plotter process to create the figure.

        Args:
            algorithm_name: name of the algorithm being run
            trial: trial number (0-indexed)
            queue: multiprocessing queue for sending data to the plotter

        """
        self._algorithm_name = algorithm_name
        self._trial = trial
        self._queue = queue
        # Use class name as metric_id so all instances of the same metric type share one figure
        self._metric_id = self.__class__.__name__

        # Send initialization message to plotter process to create figure
        # The plotter will handle deduplication (won't create duplicate figures)
        if self._queue is not None:
            with contextlib.suppress(Exception):
                self._queue.put(("init", self._metric_id, self.description, self._save_path), block=False)

    def update_plot(self, problem: "BenchmarkProblem", agents: Sequence["Agent"], iteration: int) -> None:
        """
        Update the plot with a new data point.

        Computes the metric value and sends it to the centralized plotter via queue.

        Args:
            problem: benchmark problem being solved
            agents: sequence of agents with their current state
            iteration: current iteration number

        """
        # Compute metric value
        value = self.compute(problem, agents, iteration)

        # Send data to plotter process queue
        if self._queue is not None:
            with contextlib.suppress(Exception):
                self._queue.put((self._metric_id, self._algorithm_name, self._trial, iteration, value), block=False)

    def should_update(self, iteration: int) -> bool:
        """
        Check if the metric should be updated at this iteration.

        Args:
            iteration: current iteration number

        Returns:
            True if the metric should be updated, False otherwise.

        """
        return iteration % self.update_interval == 0 or iteration == 0
