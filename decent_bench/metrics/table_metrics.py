import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
import tabulate as tb
from numpy import linalg as la
from scipy import stats

import decent_bench.metrics.metric_utils as utils
import decent_bench.utils.interoperability as iop
from decent_bench.agents import AgentMetricsView
from decent_bench.benchmark_problem import BenchmarkProblem
from decent_bench.distributed_algorithms import Algorithm
from decent_bench.utils.logger import LOGGER

Statistic = Callable[[Sequence[float]], float]


class TableMetric(ABC):
    """
    Metric to display in the statistical results table at the end of the benchmarking execution.

    Args:
        statistics: sequence of statistics such as :func:`min`, :func:`sum`, and :func:`~numpy.average` used for
            aggregating the data retrieved with :func:`get_data_from_trial` into a single value, each statistic gets its
            own row in the table
        fmt: format string used to format the values in the table, defaults to ".2e". See :meth:`str.format`
            documentation for details on the format string options.

    """

    def __init__(self, statistics: list[Statistic], fmt: str = ".2e"):
        self.statistics = statistics
        self.fmt = fmt

    @property
    @abstractmethod
    def description(self) -> str:
        """Metric description to display in the table."""

    @abstractmethod
    def get_data_from_trial(self, agents: list[AgentMetricsView], problem: BenchmarkProblem) -> Sequence[float]:
        """Extract trial data to be aggregated into a single value by each of the *statistics*."""


class Regret(TableMetric):
    """
    Global regret using the agents' final x.

    Global regret is defined as:

    .. include:: snippets/global_cost_error.rst
    """

    description: str = "regret \n[<1e-9 = exact conv.]"

    def get_data_from_trial(self, agents: list[AgentMetricsView], problem: BenchmarkProblem) -> tuple[float]:  # noqa: D102
        return (utils.regret(agents, problem, iteration=-1),)


class GradientNorm(TableMetric):
    """
    Global gradient norm using the agents' final x.

    Global gradient norm is defined as:

    .. include:: snippets/global_gradient_optimality.rst
    """

    description: str = "gradient norm"

    def get_data_from_trial(self, agents: list[AgentMetricsView], _: BenchmarkProblem) -> tuple[float]:  # noqa: D102
        return (utils.gradient_norm(agents, iteration=-1),)


class XError(TableMetric):
    r"""
    X error per agent as defined below.

    .. math::
        \{ \|\mathbf{x}_i - \mathbf{x}^\star\|, \|\mathbf{x}_j - \mathbf{x}^\star\|, ... \}

    where :math:`\mathbf{x}_i` is agent i's final x,
    :math:`\mathbf{x}_j` is agent j's final x,
    and :math:`\mathbf{x}^\star` is the optimal x defined in the *problem*.

    """

    description: str = "x error"

    def get_data_from_trial(self, agents: list[AgentMetricsView], problem: BenchmarkProblem) -> list[float]:  # noqa: D102
        return [
            float(la.norm(iop.to_numpy(problem.x_optimal) - iop.to_numpy(a.x_history[max(a.x_history)])))
            for a in agents
        ]


class AsymptoticConvergenceOrder(TableMetric):
    """
    Asymptotic convergence order per agent as defined below.

    .. include:: snippets/asymptotic_convergence_rate_and_order.rst
    """

    description: str = "asymptotic convergence order"

    def get_data_from_trial(self, agents: list[AgentMetricsView], problem: BenchmarkProblem) -> list[float]:  # noqa: D102
        return [utils.asymptotic_convergence_rate_and_order(a, problem)[1] for a in agents]


class AsymptoticConvergenceRate(TableMetric):
    """
    Asymptotic convergence rate per agent as defined below.

    .. include:: snippets/asymptotic_convergence_rate_and_order.rst
    """

    description: str = "asymptotic convergence rate"

    def get_data_from_trial(self, agents: list[AgentMetricsView], problem: BenchmarkProblem) -> list[float]:  # noqa: D102
        return [utils.asymptotic_convergence_rate_and_order(a, problem)[0] for a in agents]


class IterativeConvergenceOrder(TableMetric):
    """
    Iterative convergence order per agent as defined below.

    .. include:: snippets/iterative_convergence_rate_and_order.rst
    """

    description: str = "iterative convergence order"

    def get_data_from_trial(self, agents: list[AgentMetricsView], problem: BenchmarkProblem) -> list[float]:  # noqa: D102
        return [utils.iterative_convergence_rate_and_order(a, problem)[1] for a in agents]


class IterativeConvergenceRate(TableMetric):
    """
    Iterative convergence rate per agent as defined below.

    .. include:: snippets/iterative_convergence_rate_and_order.rst
    """

    description: str = "iterative convergence rate"

    def get_data_from_trial(self, agents: list[AgentMetricsView], problem: BenchmarkProblem) -> list[float]:  # noqa: D102
        return [utils.iterative_convergence_rate_and_order(a, problem)[0] for a in agents]


class XUpdates(TableMetric):
    """Number of iterations/updates of x per agent."""

    description: str = "nr x updates"

    def get_data_from_trial(self, agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[float]:  # noqa: D102
        return [a.x_updates for a in agents]


class FunctionCalls(TableMetric):
    """Number of cost function evaluate calls per agent."""

    description: str = "nr function calls"

    def get_data_from_trial(self, agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[float]:  # noqa: D102
        return [a.n_function_calls for a in agents]


class GradientCalls(TableMetric):
    """Number of cost function gradient calls per agent."""

    description: str = "nr gradient calls"

    def get_data_from_trial(self, agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[float]:  # noqa: D102
        return [a.n_gradient_calls for a in agents]


class HessianCalls(TableMetric):
    """Number of cost function hessian calls per agent."""

    description: str = "nr hessian calls"

    def get_data_from_trial(self, agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[float]:  # noqa: D102
        return [a.n_hessian_calls for a in agents]


class ProximalCalls(TableMetric):
    """Number of cost function proximal calls per agent."""

    description: str = "nr proximal calls"

    def get_data_from_trial(self, agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[float]:  # noqa: D102
        return [a.n_proximal_calls for a in agents]


class SentMessages(TableMetric):
    """Number of sent messages per agent."""

    description: str = "nr sent messages"

    def get_data_from_trial(self, agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[float]:  # noqa: D102
        return [a.n_sent_messages for a in agents]


class ReceivedMessages(TableMetric):
    """Number of received messages per agent."""

    description: str = "nr received messages"

    def get_data_from_trial(self, agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[float]:  # noqa: D102
        return [a.n_received_messages for a in agents]


class SentMessagesDropped(TableMetric):
    """Number of sent messages that were dropped per agent."""

    description: str = "nr sent messages dropped"

    def get_data_from_trial(self, agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[float]:  # noqa: D102
        return [a.n_sent_messages_dropped for a in agents]


DEFAULT_TABLE_METRICS = [
    Regret([utils.single]),
    GradientNorm([utils.single]),
    XError([min, np.average, max]),
    AsymptoticConvergenceOrder([np.average]),
    AsymptoticConvergenceRate([np.average]),
    IterativeConvergenceOrder([np.average]),
    IterativeConvergenceRate([np.average]),
    XUpdates([np.average, sum]),
    FunctionCalls([np.average, sum]),
    GradientCalls([np.average, sum]),
    HessianCalls([np.average, sum]),
    ProximalCalls([np.average, sum]),
    SentMessages([np.average, sum]),
    ReceivedMessages([np.average, sum]),
    SentMessagesDropped([np.average, sum]),
]
"""
- :class:`Regret` - :func:`~.metric_utils.single`
- :class:`GradientNorm` - :func:`~.metric_utils.single`
- :class:`XError` - :func:`min`, :func:`~numpy.average`, :func:`max`
- :class:`AsymptoticConvergenceOrder` - :func:`~numpy.average`
- :class:`AsymptoticConvergenceRate` - :func:`~numpy.average`
- :class:`IterativeConvergenceOrder` - :func:`~numpy.average`
- :class:`IterativeConvergenceRate` - :func:`~numpy.average`
- :class:`XUpdates` - :func:`~numpy.average`, :func:`sum`
- :class:`FunctionCalls` - :func:`~numpy.average`, :func:`sum`
- :class:`GradientCalls` - :func:`~numpy.average`, :func:`sum`
- :class:`HessianCalls` - :func:`~numpy.average`, :func:`sum`
- :class:`ProximalCalls` - :func:`~numpy.average`, :func:`sum`
- :class:`SentMessages` - :func:`~numpy.average`, :func:`sum`
- :class:`ReceivedMessages` - :func:`~numpy.average`, :func:`sum`
- :class:`SentMessagesDropped` - :func:`~numpy.average`, :func:`sum`

:meta hide-value:
"""


TABLE_METRICS_DOC_LINK = "https://decent-bench.readthedocs.io/en/latest/api/decent_bench.metrics.table_metrics.html"


def tabulate(
    resulting_agent_states: dict[Algorithm, list[list[AgentMetricsView]]],
    problem: BenchmarkProblem,
    metrics: list[TableMetric],
    confidence_level: float,
    table_fmt: Literal["grid", "latex"],
) -> None:
    """
    Print table with confidence intervals, one row per metric and statistic, and one column per algorithm.

    Args:
        resulting_agent_states: resulting agent states from the trial executions, grouped by algorithm
        problem: benchmark problem whose properties, e.g.
            :attr:`~decent_bench.benchmark_problem.BenchmarkProblem.x_optimal`,
            are used for metric calculations
        metrics: metrics to calculate
        confidence_level: confidence level of the confidence intervals
        table_fmt: table format, grid is suitable for the terminal while latex can be copy-pasted into a latex document

    """
    if not metrics:
        return
    LOGGER.info(f"Table metric definitions can be found here: {TABLE_METRICS_DOC_LINK}")
    algs = list(resulting_agent_states)
    headers = ["Metric (statistic)"] + [alg.name for alg in algs]
    rows: list[list[str]] = []
    statistics_abbr = {"average": "avg", "median": "mdn"}
    with warnings.catch_warnings(action="ignore"), utils.MetricProgressBar() as progress:
        n_statistics = sum(len(metric.statistics) for metric in metrics)
        table_task = progress.add_task("Generating table", total=n_statistics, status="")
        for metric in metrics:
            progress.update(table_task, status=f"Task: {metric.description}")
            data_per_trial = [_data_per_trial(resulting_agent_states[a], problem, metric) for a in algs]
            for statistic in metric.statistics:
                row = [f"{metric.description} ({statistics_abbr.get(statistic.__name__) or statistic.__name__})"]
                for i in range(len(algs)):
                    agg_data_per_trial = [statistic(trial) for trial in data_per_trial[i]]
                    mean, margin_of_error = _calculate_mean_and_margin_of_error(agg_data_per_trial, confidence_level)
                    formatted_confidence_interval = _format_confidence_interval(mean, margin_of_error, metric.fmt)
                    row.append(formatted_confidence_interval)
                rows.append(row)
                progress.advance(table_task)
        progress.update(table_task, status="Finalizing table")
    formatted_table = tb.tabulate(rows, headers, tablefmt=table_fmt)
    LOGGER.info("\n" + formatted_table)


def _data_per_trial(
    agents_per_trial: list[list[AgentMetricsView]], problem: BenchmarkProblem, metric: TableMetric
) -> list[Sequence[float]]:
    data_per_trial: list[Sequence[float]] = []
    for agents in agents_per_trial:
        trial_data = metric.get_data_from_trial(agents, problem)
        data_per_trial.append(trial_data)

    return data_per_trial


def _calculate_mean_and_margin_of_error(data: list[float], confidence_level: float) -> tuple[float, float]:
    mean = np.mean(data)
    sem = stats.sem(data) if len(set(data)) > 1 else None
    raw_interval = (
        stats.t.interval(confidence=confidence_level, df=len(data) - 1, loc=mean, scale=sem) if sem else (mean, mean)
    )
    if np.isfinite(mean) and np.isfinite(raw_interval).all():
        return (float(mean), float(mean - raw_interval[0]))

    return np.nan, np.nan


def _format_confidence_interval(mean: float, margin_of_error: float, fmt: str) -> str:
    try:
        formatted_confidence_interval = f"{mean:{fmt}} \u00b1 {margin_of_error:{fmt}}"
    except ValueError:
        LOGGER.warning(f"Invalid format string '{fmt}', defaulting to scientific notation")
        formatted_confidence_interval = f"{mean:.2e} \u00b1 {margin_of_error:.2e}"

    if any(np.isnan([mean, margin_of_error])):
        formatted_confidence_interval += " (diverged?)"

    return formatted_confidence_interval
