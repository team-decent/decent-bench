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

    In order to create a new metric, subclass this class and implement the abstract methods
    :func:`table_description`, :func:`plot_description`, and :func:`get_data_from_trial`.
    If you don't want the table to indicate whether the metric has diverged then override :func:`can_diverge`
    to return False. If you don't want to use the default behavior for :func:`get_table_data` or :func:`get_plot_data`
    you can also override those methods but this is not common. See the documentation for each method for more details
    and implementation specifications.

    Args:
        statistics: sequence of statistics such as :func:`min`, :func:`sum`, and :func:`~numpy.average` used for
            aggregating the data retrieved with :func:`get_data_from_trial` into a single value, each statistic gets its
            own row in the table.
        fmt: format string used to format the values in the table, defaults to ".2e". Common formats include:

            - ".2e": scientific notation with 2 decimal places
            - ".3f": fixed-point notation with 3 decimal places
            - ".4g": general format with 4 significant digits
            - ".1%": percentage format with 1 decimal place

            Where the integer specifies the precision.
            See :meth:`str.format` documentation for details on the format string options.
        x_log: whether to apply log scaling to the x-axis in plots.
        y_log: whether to apply log scaling to the y-axis in plots.
        common_iterations: whether to only plot iterations that are common to all agents. If True then only the
            intersection of all iterations reached by all agents will be plotted. If False then the union of all
            iterations reached by any agent will be plotted, and missing samples for agents that did not sample
            at those iterations will be replaced by the most recent previous sample for that agent.

    """

    def __init__(
        self,
        statistics: Sequence[Statistic],
        *,
        fmt: str = ".2e",
        x_log: bool = False,
        y_log: bool = True,
        common_iterations: bool = False,
    ) -> None:
        self.statistics = statistics
        self.x_log = x_log
        self.y_log = y_log
        self.fmt = fmt
        self._common_iterations = common_iterations

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

        If True then the table will try to indicate if the metric has diverged.
        Has no real impact on calculations of the metric, will not affect plots.
        """
        return True

    @property
    def common_iterations(self) -> bool:
        """
        Whether to only plot iterations that are common to all agents.

        If True then only the intersection of all iterations reached by all agents will be plotted.
        If False then the union of all iterations reached by any agent will be plotted, and missing samples for
        agents that did not sample at those iterations will be replaced by the most recent previous sample for that
        agent.

        This only affects which iterations are plotted, it does not affect the data that is shown in the table or
        whether the metric is calculated at the last iteration or not.
        """
        return self._common_iterations

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

        Note:
            If :attr:`~decent_bench.metrics.Metric.common_iterations` is True then *iteration* will be one of the
            iterations that any agents has reached, therefore it might not be present in all agents' x_history.
            In this case, the implementation of this method should handle this scenario appropriately by calling
            :func:`~decent_bench.metrics.metric_utils.find_closest_iteration`.
            If :attr:`~decent_bench.metrics.Metric.common_iterations` is False then *iteration* will be one of the
            iterations that all agents have reached, so it should be present in all agents' x_history but the
            implementation should be able to handle both cases.

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
        By default, it calculates statistics (mean) on the intersection of all the iterations
        reached by all agents, but it can be overridden to perform additional
        processing on the data before it is used in plots.
        """
        iterations = (
            utils.common_sorted_iterations(agents) if self.common_iterations else utils.all_sorted_iterations(agents)
        )
        return [(i, float(np.mean(self.get_data_from_trial(agents, problem, i)))) for i in iterations]
