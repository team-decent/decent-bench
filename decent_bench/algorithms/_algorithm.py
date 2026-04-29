from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, final

from decent_bench.networks import Network

if TYPE_CHECKING:
    from decent_bench.agents import Agent


class Algorithm[NetworkT: Network](ABC):
    """Base class for decentralized algorithms."""

    def __post_init__(self) -> None:
        """Optional hook to be called by dataclasses after __init__."""  # noqa: D401
        return

    def __init_subclass__(cls, **kwargs: dict[str, Any]) -> None:
        """Validate `iterations` for all subclasses."""
        super().__init_subclass__(**kwargs)

        # override __post_init__ to inject `iterations` validation
        original_post_init: Callable[[Algorithm[NetworkT]], None] | None = getattr(cls, "__post_init__", None)

        def __post_init__(self: "Algorithm[NetworkT]") -> None:  # noqa: N807
            # inject `iterations` validation
            if self.iterations <= 0:
                raise ValueError("`iterations` must be positive")

            # add subclass's __post_init__ if any
            if original_post_init:
                original_post_init(self)

        setattr(cls, "__post_init__", __post_init__)  # noqa: B010

    iterations: int
    """Number of iterations to run the algorithm for."""

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

    @abstractmethod
    def cleanup_agents(self, network: NetworkT) -> Iterable["Agent"]:
        """
        Return the agents whose auxiliary variables should be cleared.

        Args:
            network: provides the agents and topology for this algorithm.

        """

    def cleanup(self, network: NetworkT) -> None:
        """
        Clean up the algorithm state by clearing auxiliary variables from agents.

        This method is used to free up memory used by auxiliary variables that are not needed after training.
        Can be overridden to control what gets cleaned up.

        Note:
            Override :meth:`~decent_bench.algorithms.Algorithm.cleanup_agents` to control which
            agents are cleaned up.

        Args:
            network: provides the agents and topology for this algorithm.

        """
        for agent in self.cleanup_agents(network):
            if agent.aux_vars is not None:
                agent.aux_vars.clear()

    @final
    def _snapshot_agents(self, network: NetworkT, iteration: int) -> None:
        for i in network.agents():
            # Forcefully save a snapshot on the final iteration
            i._snapshot(iteration=iteration, force=iteration == self.iterations)  # noqa: SLF001

    @final
    def run(
        self,
        network: NetworkT,
        start_iteration: int = 0,
        progress_callback: Callable[[int], None] | None = None,
    ) -> None:
        """
        Run the algorithm.

        This method first calls :meth:`initialize`, then :meth:`step` for the specified number of iterations.
        Optionally call :meth:`cleanup` after :meth:`run` to clear auxiliary variables
        and free up memory.

        Args:
            network: provides the agents and topology for this algorithm.
            start_iteration: iteration number to start from, used when resuming from a checkpoint. If greater than 0,
                :meth:`initialize` will be skipped.
            progress_callback: optional callback to report progress after each iteration.

        Raises:
            ValueError: if start_iteration is not in [0, iterations]

        Warning:
            Do not override this method. Instead, override :meth:`initialize` and :meth:`step` as needed.

        Note:
            The algorithm saves the agents' states every :attr:`~decent_bench.agents.Agent.state_snapshot_period`.

        """
        if start_iteration < 0 or start_iteration > self.iterations:
            raise ValueError(
                f"Invalid start_iteration {start_iteration} for algorithm with {self.iterations} iterations"
            )

        if start_iteration == 0:
            self.initialize(network)

        for k in range(start_iteration, self.iterations):
            network._step(k)  # noqa: SLF001
            self.step(network, k)
            # Already completed the iteration, so snapshot with k+1 to indicate the state after iteration k
            self._snapshot_agents(network, k + 1)
            if progress_callback is not None:
                progress_callback(k)
