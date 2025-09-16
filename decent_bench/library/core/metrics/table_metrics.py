from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy import linalg as la

import decent_bench.library.core.metrics.metric_utils as utils
from decent_bench.library.core.agent import AgentMetricsView
from decent_bench.library.core.benchmark_problem.benchmark_problems import BenchmarkProblem


def global_cost_error(agents: list[AgentMetricsView], problem: BenchmarkProblem) -> tuple[float]:
    """
    Calculate the global cost error using the agents' final x.

    For details, see :func:`~decent_bench.library.core.metrics.metric_utils.global_cost_error_at_iter`.
    """
    return (utils.global_cost_error_at_iter(agents, problem, iteration=-1),)


def global_gradient_optimality(agents: list[AgentMetricsView], _: BenchmarkProblem) -> tuple[float]:
    """
    Calculate the global gradient optimality using the agents' final x.

    For details, see :func:`~decent_bench.library.core.metrics.metric_utils.global_gradient_optimality_at_iter`.
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
    Calculate the asymptotic convergence rate per agent.

    For details, see :func:`~decent_bench.library.core.metrics.metric_utils.asymptotic_convergence_rate_and_order`.
    """
    return [utils.asymptotic_convergence_rate_and_order(a, problem)[0] for a in agents]


def asymptotic_convergence_order(agents: list[AgentMetricsView], problem: BenchmarkProblem) -> list[float]:
    """
    Calculate the asymptotic convergence order per agent.

    For details, see :func:`~decent_bench.library.core.metrics.metric_utils.asymptotic_convergence_rate_and_order`.
    """
    return [utils.asymptotic_convergence_rate_and_order(a, problem)[1] for a in agents]


def iterative_convergence_rate(agents: list[AgentMetricsView], problem: BenchmarkProblem) -> list[float]:
    """
    Calculate the iterative convergence rate per agent.

    For details, see :func:`~decent_bench.library.core.metrics.metric_utils.iterative_convergence_rate_and_order`.
    """
    return [utils.iterative_convergence_rate_and_order(a, problem)[0] for a in agents]


def iterative_convergence_order(agents: list[AgentMetricsView], problem: BenchmarkProblem) -> list[float]:
    """
    Calculate the iterative convergence order per agent.

    For details, see :func:`~decent_bench.library.core.metrics.metric_utils.iterative_convergence_rate_and_order`.
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
Assert that only one value exists and return it using :func:`~decent_bench.library.core.metrics.metric_utils.single`.
"""

DEFAULT_TABLE_METRICS = [
    TableMetric("global cost error", [Single], global_cost_error),
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
- Global cost error (single) defined at :func:`global_cost_error`.
- Global gradient optimality (single) defined at :func:`global_gradient_optimality`.
- x error (min, avg, max) defined at :func:`x_error`.
- Asymptotic convergence order (avg) defined at :func:`asymptotic_convergence_order`.
- Asymptotic convergence rate (avg) defined at :func:`asymptotic_convergence_rate`.
- Iterative convergence order (avg) defined at :func:`iterative_convergence_order`.
- Iterative convergence rate (avg) defined at :func:`iterative_convergence_rate`.
- Nr of x updates (avg, sum) defined at :func:`n_x_updates`.
- Nr of evaluate calls (avg, sum) defined at :func:`n_evaluate_calls`.
- Nr of gradient calls (avg, sum) defined at :func:`n_gradient_calls`.
- Nr of hessian calls (avg, sum) defined at :func:`n_hessian_calls`.
- Nr of proximal calls (avg, sum) defined at :func:`n_proximal_calls`.
- Nr of sent messages (avg, sum) defined at :func:`n_sent_messages`.
- Nr of received messages (avg, sum) defined at :func:`n_received_messages`.
- Nr of sent messages dropped (avg, sum) defined at :func:`n_sent_messages_dropped`.

:meta hide-value:
"""
