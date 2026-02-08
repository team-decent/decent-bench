from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import numpy as np

import decent_bench.metrics.metric_utils as utils
from decent_bench.agents import AgentMetricsView
from decent_bench.benchmark_problem import BenchmarkProblem

Statistic = Callable[[Sequence[float]], float]

X = float
"""Type alias for the x values in plots, typically iterations or time."""

Y = float
"""Type alias for the y values in plots, typically the metric value."""


class Metric(ABC):
    """
    Abstract base class for metrics.

    Args:
        statistics: sequence of statistics such as :func:`min`, :func:`sum`, and :func:`~numpy.average` used for
            aggregating the data retrieved with :func:`get_data_from_trial` into a single value, each statistic gets its
            own row in the table
        x_log: whether to apply log scaling to the x-axis in plots.
        y_log: whether to apply log scaling to the y-axis in plots.
        fmt: format string used to format the values in the table, defaults to ".2e". Common formats include:
            - ".2e": scientific notation with 2 decimal places
            - ".3f": fixed-point notation with 3 decimal places
            - ".4g": general format with 4 significant digits
            - ".1%": percentage format with 1 decimal place

            Where the integer specifies the precision.
            See :meth:`str.format` documentation for details on the format string options.

    """

    def __init__(
        self,
        statistics: Sequence[Statistic],
        *,
        fmt: str = ".2e",
        x_log: bool = False,
        y_log: bool = True,
    ) -> None:
        self.statistics = statistics
        self.x_log = x_log
        self.y_log = y_log
        self.fmt = fmt

    @property
    @abstractmethod
    def table_description(self) -> str:
        """Metric description to display in the table."""

    @property
    @abstractmethod
    def plot_description(self) -> str:
        """Label for the y-axis in plots."""

    @property
    def can_diverge(self) -> bool:
        """
        Indicates whether the metric can diverge, i.e. take on infinite or NaN values.

        If True then the table will try to indicate if the has metric diverged.
        Has no real impact on calulations of the metric, will not effect plots.
        """
        return True

    @abstractmethod
    def get_data_from_trial(
        self,
        agents: Sequence[AgentMetricsView],
        problem: BenchmarkProblem,
        iteration: int,
    ) -> Sequence[float]:
        """
        Get the data for this metric from a trial.

        Args:
            agents: the agents being evaluated
            problem: the benchmark problem being evaluated
            iteration: the iteration at which to evaluate the metric, or -1 to use the agents' final x

        Returns:
            a list of floats, one for each agent

        """

    def get_table_data(self, agents: Sequence[AgentMetricsView], problem: BenchmarkProblem) -> Sequence[float]:
        """
        Extract trial data to be used in the table for this metric.

        This is used by :func:`~decent_bench.metrics.create_tables` to generate the table for this metric.
        By default, it returns the metric from the last iteration,
        but it can be overridden to perform additional processing on the data before it is used in the table.
        """
        return self.get_data_from_trial(agents, problem, -1)

    def get_plot_data(self, agents: Sequence[AgentMetricsView], problem: BenchmarkProblem) -> Sequence[tuple[X, Y]]:
        """
        Extract trial data to be used in plots for this metric.

        This is used by :func:`~decent_bench.metrics.create_plots` to generate plots for this metric.
        By default, it calculates statistics on the intersection of all the iterations
        reached by all agents, but it can be overridden to perform additional
        processing on the data before it is used in plots.
        """
        return [
            (i, float(np.mean(self.get_data_from_trial(agents, problem, i))))
            for i in utils.common_sorted_iterations(agents)
        ]
