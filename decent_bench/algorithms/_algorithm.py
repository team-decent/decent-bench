from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import final

from decent_bench.networks import Network


class Algorithm[NetworkT: Network](ABC):
    """Base class for decentralized algorithms."""

    def __post_init__(self) -> None:
        """Optional hook to be called by dataclasses after __init__."""  # noqa: D401
        return

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the algorithm."""

    @abstractmethod
    def initialize(self, network: NetworkT) -> None:
        """
        Initialize the algorithm.

        Args:
            network: provides the agents and topology for this algorithm.

        """

    @abstractmethod
    def step(self, network: NetworkT, iteration: int) -> None:
        """
        Perform one iteration of the algorithm.

        Args:
            network: provides the agents and topology for this algorithm.
            iteration: current iteration number.

        """

    def _cleanup(self, network: NetworkT) -> None:
        """
        Clean up the algorithm state by clearing auxiliary variables from agents.

        This method is used to free up memory used by auxiliary variables that are not needed after training.
        Can be overridden to control what gets cleaned up.

        Args:
            network: provides the agents and topology for this algorithm.

        """
        for agent in network.graph.nodes():
            if agent.aux_vars is not None:
                agent.aux_vars.clear()

    @final
    def _snapshot_agents(self, network: NetworkT, iteration: int, iterations: int) -> None:
        for i in network.snapshot_agents():
            # Forcefully save a snapshot on the final iteration
            i._snapshot(iteration=iteration, force=iteration == iterations)  # noqa: SLF001

    @final
    def run(
        self,
        network: NetworkT,
        iterations: int,
        start_iteration: int = 0,
        progress_callback: Callable[[int], None] | None = None,
    ) -> None:
        """
        Run the algorithm.

        This method first calls :meth:`initialize`, then :meth:`step` for the specified number of iterations.

        Args:
            network: provides the agents and topology for this algorithm.
            iterations: total number of iterations to run.
            start_iteration: iteration number to start from, used when resuming from a checkpoint. If greater than 0,
                :meth:`initialize` will be skipped.
            progress_callback: optional callback to report progress after each iteration.

        Raises:
            ValueError: if iterations is not positive or start_iteration is not in [0, iterations]

        Warning:
            Do not override this method. Instead, override :meth:`initialize` and :meth:`step` as needed.

        Note:
            The algorithm saves the agents' states every :attr:`~decent_bench.agents.Agent.state_snapshot_period`.
            The final state is always snapshotted even when it falls outside the configured snapshot period.

        """
        if iterations <= 0:
            raise ValueError("`iterations` must be positive")
        if start_iteration < 0 or start_iteration > iterations:
            raise ValueError(f"Invalid start_iteration {start_iteration} for algorithm with {iterations} iterations")

        if start_iteration == 0:
            self.initialize(network)

        for k in range(start_iteration, iterations):
            network._step(k)  # noqa: SLF001
            self.step(network, k)
            # Already completed the iteration, so snapshot with k+1 to indicate the state after iteration k
            self._snapshot_agents(network, k + 1, iterations)
            if progress_callback is not None:
                progress_callback(k)
