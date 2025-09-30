import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
import tabulate as tb
from numpy import linalg as la
from scipy import stats

import decent_bench.metrics.metric_utils as utils
from decent_bench.agent import AgentMetricsView
from decent_bench.benchmark_problem import BenchmarkProblem
from decent_bench.distributed_algorithms import DstAlgorithm
from decent_bench.utils.logger import LOGGER

Statistic = Callable[[Sequence[float]], float]


class TableMetric(ABC):
    """
    Metric to display in the statistical results table at the end of the benchmarking execution.

    Args:
        statistics: sequence of statistics such as :func:`min`, :func:`sum`, and :func:`~numpy.average` used for
            aggregating the data retrieved with :func:`get_data_from_trial` into a single value, each statistic gets its
            own row in the table

    """

    def __init__(self, statistics: list[Statistic]):
        self.statistics = statistics

    @property
    @abstractmethod
    def description(self) -> str:
        """Metric description to display in the table."""

    @abstractmethod
    def get_data_from_trial(self, agents: list[AgentMetricsView], problem: BenchmarkProblem) -> Sequence[float]:
        """Extract trial data to be aggregated into a single value by each of the *statistics*."""


class GlobalCostError(TableMetric):
    """
    Global cost error using the agents' final x.

    Global cost error is defined as:

    .. include:: snippets/global_cost_error.rst
    """

    description: str = "global cost error \n[<1e-9 = exact conv.]"

    def get_data_from_trial(self, agents: list[AgentMetricsView], problem: BenchmarkProblem) -> tuple[float]:  # noqa: D102
        return (utils.global_cost_error_at_iter(agents, problem, iteration=-1),)


class GlobalGradientOptimality(TableMetric):
    """
    Global gradient optimality using the agents' final x.

    Global gradient optimality is defined as:

    .. include:: snippets/global_gradient_optimality.rst
    """

    description: str = "global gradient optimality"

    def get_data_from_trial(self, agents: list[AgentMetricsView], _: BenchmarkProblem) -> tuple[float]:  # noqa: D102
        return (utils.global_gradient_optimality_at_iter(agents, iteration=-1),)


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
        return [float(la.norm(problem.optimal_x - a.x_per_iteration[-1])) for a in agents]


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


class NrXUpdates(TableMetric):
    """Number of iterations/updates of x per agent."""

    description: str = "nr x updates"

    def get_data_from_trial(self, agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[float]:  # noqa: D102
        return [len(a.x_per_iteration) - 1 for a in agents]


class NrEvaluateCalls(TableMetric):
    """Number of cost function evaluate calls per agent."""

    description: str = "nr evaluate calls"

    def get_data_from_trial(self, agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[float]:  # noqa: D102
        return [a.n_evaluate_calls for a in agents]


class NrGradientCalls(TableMetric):
    """Number of cost function gradient calls per agent."""

    description: str = "nr gradient calls"

    def get_data_from_trial(self, agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[float]:  # noqa: D102
        return [a.n_gradient_calls for a in agents]


class NrHessianCalls(TableMetric):
    """Number of cost function hessian calls per agent."""

    description: str = "nr hessian calls"

    def get_data_from_trial(self, agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[float]:  # noqa: D102
        return [a.n_hessian_calls for a in agents]


class NrProximalCalls(TableMetric):
    """Number of cost function proximal calls per agent."""

    description: str = "nr proximal calls"

    def get_data_from_trial(self, agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[float]:  # noqa: D102
        return [a.n_proximal_calls for a in agents]


class NrSentMessages(TableMetric):
    """Number of sent messages per agent."""

    description: str = "nr sent messages"

    def get_data_from_trial(self, agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[float]:  # noqa: D102
        return [a.n_sent_messages for a in agents]


class NrReceivedMessages(TableMetric):
    """Number of received messages per agent."""

    description: str = "nr received messages"

    def get_data_from_trial(self, agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[float]:  # noqa: D102
        return [a.n_received_messages for a in agents]


class NrSentMessagesDropped(TableMetric):
    """Number of sent messages that were dropped per agent."""

    description: str = "nr sent messages dropped"

    def get_data_from_trial(self, agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[float]:  # noqa: D102
        return [a.n_sent_messages_dropped for a in agents]


DEFAULT_TABLE_METRICS = [
    GlobalCostError([utils.single]),
    GlobalGradientOptimality([utils.single]),
    XError([min, np.average, max]),
    AsymptoticConvergenceOrder([np.average]),
    AsymptoticConvergenceRate([np.average]),
    IterativeConvergenceOrder([np.average]),
    IterativeConvergenceRate([np.average]),
    NrXUpdates([np.average, sum]),
    NrEvaluateCalls([np.average, sum]),
    NrGradientCalls([np.average, sum]),
    NrHessianCalls([np.average, sum]),
    NrProximalCalls([np.average, sum]),
    NrSentMessages([np.average, sum]),
    NrReceivedMessages([np.average, sum]),
    NrSentMessagesDropped([np.average, sum]),
]
"""
- :class:`GlobalCostError` - :func:`~.metric_utils.single`
- :class:`GlobalGradientOptimality` - :func:`~.metric_utils.single`
- :class:`XError` - :func:`min`, :func:`~numpy.average`, :func:`max`
- :class:`AsymptoticConvergenceOrder` - :func:`~numpy.average`
- :class:`AsymptoticConvergenceRate` - :func:`~numpy.average`
- :class:`IterativeConvergenceOrder` - :func:`~numpy.average`
- :class:`IterativeConvergenceRate` - :func:`~numpy.average`
- :class:`NrXUpdates` - :func:`~numpy.average`, :func:`sum`
- :class:`NrEvaluateCalls` - :func:`~numpy.average`, :func:`sum`
- :class:`NrGradientCalls` - :func:`~numpy.average`, :func:`sum`
- :class:`NrHessianCalls` - :func:`~numpy.average`, :func:`sum`
- :class:`NrProximalCalls` - :func:`~numpy.average`, :func:`sum`
- :class:`NrSentMessages` - :func:`~numpy.average`, :func:`sum`
- :class:`NrReceivedMessages` - :func:`~numpy.average`, :func:`sum`
- :class:`NrSentMessagesDropped` - :func:`~numpy.average`, :func:`sum`

:meta hide-value:
"""


TABLE_METRICS_DOC_LINK = "https://decent-bench.readthedocs.io/en/latest/api/decent_bench.metrics.table_metrics.html"


def tabulate(
    resulting_agent_states: dict[DstAlgorithm, list[list[AgentMetricsView]]],
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
            :attr:`~decent_bench.benchmark_problem.BenchmarkProblem.optimal_x`,
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
    for metric in metrics:
        for statistic in metric.statistics:
            row = [f"{metric.description} ({statistics_abbr.get(statistic.__name__) or statistic.__name__})"]
            for alg in algs:
                agent_states_per_trial = resulting_agent_states[alg]
                with warnings.catch_warnings(action="ignore"):
                    agg_data_per_trial = _aggregate_data_per_trial(agent_states_per_trial, problem, metric, statistic)
                    mean, margin_of_error = _calculate_mean_and_margin_of_error(agg_data_per_trial, confidence_level)
                formatted_confidence_interval = _format_confidence_interval(mean, margin_of_error)
                row.append(formatted_confidence_interval)
            rows.append(row)
    formatted_table = tb.tabulate(rows, headers, tablefmt=table_fmt)
    LOGGER.info("\n" + formatted_table)


def _aggregate_data_per_trial(
    agents_per_trial: list[list[AgentMetricsView]], problem: BenchmarkProblem, metric: TableMetric, statistic: Statistic
) -> list[float]:
    aggregated_data_per_trial: list[float] = []
    for agents in agents_per_trial:
        trial_data = metric.get_data_from_trial(agents, problem)
        aggregated_trial_data = statistic(trial_data)
        aggregated_data_per_trial.append(aggregated_trial_data)
    return aggregated_data_per_trial


def _calculate_mean_and_margin_of_error(data: list[float], confidence_level: float) -> tuple[float, float]:
    mean = np.mean(data)
    sem = stats.sem(data) if len(set(data)) > 1 else None
    raw_interval = (
        stats.t.interval(confidence=confidence_level, df=len(data) - 1, loc=mean, scale=sem) if sem else (mean, mean)
    )
    if np.isfinite(mean) and np.isfinite(raw_interval).all():
        return (float(mean), float(mean - raw_interval[0]))
    return np.nan, np.nan


def _format_confidence_interval(mean: float, margin_of_error: float) -> str:
    formatted_confidence_interval = f"{mean:.2e} \u00b1 {margin_of_error:.2e}"
    if any(np.isnan([mean, margin_of_error])):
        formatted_confidence_interval += " (diverged?)"
    return formatted_confidence_interval
