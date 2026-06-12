from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from decent_bench.metrics._metrics_view import NetworkMetricsView

if TYPE_CHECKING:
    from decent_bench.benchmark import BenchmarkProblem


Statistic = Callable[[Sequence[float]], float]


class Metric(ABC):
    """
    Abstract base class for metrics.

    In order to create a new metric, subclass this class and implement the abstract methods
    :func:`description` and :func:`compute`.

    Args:
        fmt: format string used to format the values in the table, defaults to ".2e". Common formats include:

            - ".2e": scientific notation with 2 decimal places
            - ".3f": fixed-point notation with 3 decimal places
            - ".4g": general format with 4 significant digits
            - ".1%": percentage format with 1 decimal place

            Where the integer specifies the precision.
            See :meth:`str.format` documentation for details on the format string options.
        x_log: whether to apply log scaling to the x-axis in plots.
        y_log: whether to apply log scaling to the y-axis in plots.

    """

    def __init__(
        self,
        fmt: str = ".2e",
        x_log: bool = False,
        y_log: bool = True,
    ) -> None:
        self.x_log = x_log
        self.y_log = y_log
        self.fmt = fmt

    @property
    @abstractmethod
    def description(self) -> str:
        """Metric description used as the table row label and y-axis label in plots."""

    def is_available(
        self,
        problem: "BenchmarkProblem",  # noqa: ARG002
    ) -> tuple[bool, str | None]:
        """
        Check whether this metric can be computed for the given problem.

        Override in subclasses that have availability preconditions (e.g. requiring
        ``problem.x_optimal`` or ``problem.test_data``). The default implementation
        always returns available.

        Args:
            problem: the benchmark problem being evaluated

        Returns:
            A tuple ``(available, reason)``. When *available* is ``True``, *reason* is
            ``None``. When *available* is ``False``, *reason* contains a human-readable
            explanation.

        """
        return True, None

    @abstractmethod
    def compute(
        self,
        network: NetworkMetricsView,
        problem: "BenchmarkProblem",
        iteration: int,
    ) -> Sequence[float]:
        """
        Evaluate the metric on the results of a trial.

        Args:
            network: the snapshotted network view being evaluated.
            problem: the benchmark problem being evaluated
            iteration: the iteration at which to compute the metric, or -1 to use the agents' final x

        Returns:
            a sequence of metric values

        """
