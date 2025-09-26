import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import tabulate as tb
from numpy import linalg as la
from scipy import stats

import decent_bench.metrics.metric_utils as utils
from decent_bench.agent import AgentMetricsView
from decent_bench.benchmark_problem import BenchmarkProblem
from decent_bench.distributed_algorithms import DstAlgorithm
from decent_bench.network import Network
from decent_bench.utils.logger import LOGGER


@dataclass(eq=False)
class Statistic:
    """
    Statistic used for aggregating multiple data points into a single value.

    Args:
        name: statistic name to display in the table
        agg_func: function that aggregates a sequence of data points into a single value

    """

    name: str
    agg_func: Callable[[Sequence[float]], float]


Min = Statistic("min", min)
"""Take the minimum using :func:`min`."""

Avg = Statistic("avg", np.average)
"""Take the average using :func:`numpy.average`."""

Mdn = Statistic("mdn", np.median)
"""Take the median using :func:`numpy.median`."""

Max = Statistic("max", max)
"""Take the maximum using :func:`max`."""

Sum = Statistic("sum", sum)
"""Take the sum using :func:`sum`."""

Single = Statistic("single", utils.single)
"""
Assert that only one value exists and return it using :func:`~decent_bench.metrics.metric_utils.single`.
"""


@dataclass(eq=False)
class TableMetric:
    """
    Metric for the statistical results table displayed at the end of the benchmarking execution.

    Args:
        name: metric name to display in the table
        statistics: sequence of statistics such as :data:`Min`, :data:`Sum`, or :data:`Avg` used for aggregating the
            data retrieved with *get_data_from_trial* into a single value, each statistic gets its own row in
            the table
        get_data_from_trial: function that takes trial data as input and extracts data to be aggregated and displayed in
            the table

    """

    name: str
    statistics: Sequence[Statistic]
    get_data_from_trial: Callable[[list[AgentMetricsView], BenchmarkProblem], Sequence[float]]


def global_cost_error(agents: list[AgentMetricsView], problem: BenchmarkProblem) -> tuple[float]:
    """
    Calculate the global cost error using the agents' final x.

    Global cost error is defined as:

    .. include:: snippets/global_cost_error.rst
    """
    return (utils.global_cost_error_at_iter(agents, problem, iteration=-1),)


def global_gradient_optimality(agents: list[AgentMetricsView], _: BenchmarkProblem) -> tuple[float]:
    """
    Calculate the global gradient optimality using the agents' final x.

    Global gradient optimality is defined as:

    .. include:: snippets/global_gradient_optimality.rst
    """
    return (utils.global_gradient_optimality_at_iter(agents, iteration=-1),)


def x_error(agents: list[AgentMetricsView], problem: BenchmarkProblem) -> list[float]:
    r"""
    Calculate the x error per agent as defined below.

    .. math::
        \{ \|\mathbf{x}_i - \mathbf{x}^\star\|, \|\mathbf{x}_j - \mathbf{x}^\star\|, ... \}

    where :math:`\mathbf{x}_i` is agent i's final x,
    :math:`\mathbf{x}_j` is agent j's final x,
    and :math:`\mathbf{x}^\star` is the optimal x defined in the *problem*.

    """
    return [float(la.norm(problem.optimal_x - a.x_per_iteration[-1])) for a in agents]


def asymptotic_convergence_rate(agents: list[AgentMetricsView], problem: BenchmarkProblem) -> list[float]:
    """
    Estimate the asymptotic convergence rate per agent as defined below.

    .. include:: snippets/asymptotic_convergence_rate_and_order.rst
    """
    return [utils.asymptotic_convergence_rate_and_order(a, problem)[0] for a in agents]


def asymptotic_convergence_order(agents: list[AgentMetricsView], problem: BenchmarkProblem) -> list[float]:
    """
    Estimate the asymptotic convergence order per agent as defined below.

    .. include:: snippets/asymptotic_convergence_rate_and_order.rst
    """
    return [utils.asymptotic_convergence_rate_and_order(a, problem)[1] for a in agents]


def iterative_convergence_rate(agents: list[AgentMetricsView], problem: BenchmarkProblem) -> list[float]:
    """
    Estimate the iterative convergence rate per agent as defined below.

    .. include:: snippets/iterative_convergence_rate_and_order.rst
    """
    return [utils.iterative_convergence_rate_and_order(a, problem)[0] for a in agents]


def iterative_convergence_order(agents: list[AgentMetricsView], problem: BenchmarkProblem) -> list[float]:
    """
    Estimate the iterative convergence order per agent as defined below.

    .. include:: snippets/iterative_convergence_rate_and_order.rst
    """
    return [utils.iterative_convergence_rate_and_order(a, problem)[1] for a in agents]


def n_x_updates(agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[float]:
    """Get the number of iterations/updates of x per agent."""
    return [len(a.x_per_iteration) - 1 for a in agents]


def n_evaluate_calls(agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[float]:
    """Get the number of cost function evaluate calls per agent."""
    return [a.n_evaluate_calls for a in agents]


def n_gradient_calls(agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[float]:
    """Get the number of cost function gradient calls per agent."""
    return [a.n_gradient_calls for a in agents]


def n_hessian_calls(agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[float]:
    """Get the number of cost function hessian calls per agent."""
    return [a.n_hessian_calls for a in agents]


def n_proximal_calls(agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[float]:
    """Get the number of cost function proximal calls per agent."""
    return [a.n_proximal_calls for a in agents]


def n_sent_messages(agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[float]:
    """Get the number of sent messages per agent."""
    return [a.n_sent_messages for a in agents]


def n_received_messages(agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[float]:
    """Get the number of received messages per agent."""
    return [a.n_received_messages for a in agents]


def n_sent_messages_dropped(agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[float]:
    """Get the number of sent messages that were dropped per agent."""
    return [a.n_sent_messages_dropped for a in agents]


DEFAULT_TABLE_METRICS = [
    TableMetric("global cost error (< 1e-9 = exact convergence)", [Single], global_cost_error),
    TableMetric("global gradient optimality", [Single], global_gradient_optimality),
    TableMetric("x error", [Min, Avg, Max], x_error),
    TableMetric("asymptotic convergence order", [Avg], asymptotic_convergence_order),
    TableMetric("asymptotic convergence rate", [Avg], asymptotic_convergence_rate),
    TableMetric("iterative convergence order", [Avg], iterative_convergence_order),
    TableMetric("iterative convergence rate", [Avg], iterative_convergence_rate),
    TableMetric("nr x updates", [Avg, Sum], n_x_updates),
    TableMetric("nr evaluate calls", [Avg, Sum], n_evaluate_calls),
    TableMetric("nr gradient calls", [Avg, Sum], n_gradient_calls),
    TableMetric("nr hessian calls", [Avg, Sum], n_hessian_calls),
    TableMetric("nr proximal calls", [Avg, Sum], n_proximal_calls),
    TableMetric("nr sent messages", [Avg, Sum], n_sent_messages),
    TableMetric("nr received messages", [Avg, Sum], n_received_messages),
    TableMetric("nr sent messages dropped", [Avg, Sum], n_sent_messages_dropped),
]
"""
- :func:`global_cost_error` (single)
- :func:`global_gradient_optimality` (single)
- :func:`x_error` (min, avg, max)
- :func:`asymptotic_convergence_order` (avg)
- :func:`asymptotic_convergence_rate` (avg)
- :func:`iterative_convergence_order` (avg)
- :func:`iterative_convergence_rate` (avg)
- :func:`n_x_updates` (avg, sum)
- :func:`n_evaluate_calls` (avg, sum)
- :func:`n_gradient_calls` (avg, sum)
- :func:`n_hessian_calls` (avg, sum)
- :func:`n_proximal_calls` (avg, sum)
- :func:`n_sent_messages` (avg, sum)
- :func:`n_received_messages` (avg, sum)
- :func:`n_sent_messages_dropped` (avg, sum)


:meta hide-value:
"""


DOC_LINK = "https://decent-bench.readthedocs.io/en/latest/decent_bench.metrics.table_metrics.html"


def tabulate(
    resulting_nw_states_per_alg: dict[DstAlgorithm, list[Network]],
    problem: BenchmarkProblem,
    metrics: list[TableMetric],
    confidence_level: float,
    table_fmt: Literal["grid", "latex"],
) -> None:
    """
    Print table with confidence intervals, one row per metric and statistic, and one column per algorithm.

    Args:
        resulting_nw_states_per_alg: resulting network states from the trial executions, grouped by algorithm
        problem: benchmark problem whose properties, e.g.
            :attr:`~decent_bench.benchmark_problem.BenchmarkProblem.optimal_x`,
            are used for metric calculations
        metrics: metrics to calculate
        confidence_level: confidence level of the confidence intervals
        table_fmt: table format, grid is suitable for the terminal while latex can be copy-pasted into a latex document

    """
    headers = ["Metric (statistic)"] + [alg.name for alg in resulting_nw_states_per_alg]
    rows: list[list[str]] = []
    for metric in metrics:
        for statistic in metric.statistics:
            row = [f"{metric.name} ({statistic.name})"]
            for nw_states in resulting_nw_states_per_alg.values():
                aggregated_data_per_trial = _get_aggregated_data_per_trial(nw_states, problem, metric, statistic)
                mean, margin_of_error = _get_mean_and_margin_of_error(aggregated_data_per_trial, confidence_level)
                formatted_confidence_interval = _format_confidence_interval(mean, margin_of_error)
                row.append(formatted_confidence_interval)
            rows.append(row)
    formatted_table = tb.tabulate(rows, headers, tablefmt=table_fmt)
    LOGGER.info("\n" + formatted_table)
    LOGGER.info(f"Metric definitions can be found here: {DOC_LINK}")


def _get_aggregated_data_per_trial(
    resulting_nw_states: list[Network], problem: BenchmarkProblem, metric: TableMetric, statistic: Statistic
) -> list[float]:
    aggregated_data_per_trial: list[float] = []
    for nw in resulting_nw_states:
        agent_metrics_views = [AgentMetricsView.from_agent(a) for a in nw.get_all_agents()]
        with warnings.catch_warnings(action="ignore"):
            trial_data = metric.get_data_from_trial(agent_metrics_views, problem)
            aggregated_trial_data = statistic.agg_func(trial_data)
        aggregated_data_per_trial.append(aggregated_trial_data)
    return aggregated_data_per_trial


def _get_mean_and_margin_of_error(data: list[float], confidence_level: float) -> tuple[float, float]:
    mean = np.mean(data)
    sem = stats.sem(data) if len(set(data)) > 1 else None
    raw_interval = (
        stats.t.interval(confidence=confidence_level, df=len(data) - 1, loc=mean, scale=sem) if sem else (mean, mean)
    )
    res = (float(mean), float(mean - raw_interval[0]))
    if any(np.isinf(res)):
        return np.nan, np.nan
    return res


def _format_confidence_interval(mean: float, margin_of_error: float) -> str:
    formatted_confidence_interval = f"{mean:.3e} \u00b1 {margin_of_error:.3e}"
    if any(np.isnan([mean, margin_of_error])):
        formatted_confidence_interval += " (diverged?)"
    return formatted_confidence_interval
