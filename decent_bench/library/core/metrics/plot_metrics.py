from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import decent_bench.library.core.metrics.metric_utils as utils
from decent_bench.library.core.agent import AgentMetricsView
from decent_bench.library.core.benchmark_problem.benchmark_problems import BenchmarkProblem


def global_cost_error_per_iteration(agents: list[AgentMetricsView], problem: BenchmarkProblem) -> list[tuple[X, Y]]:
    r"""
    Calculate the global cost error (y-axis) for each iteration (x-axis).

    Global cost error is defined at :func:`~decent_bench.library.core.metrics.metric_utils.global_cost_error_at_iter`.

    All iterations up to and including the last one reached by all *agents* are taken into account, subsequent
    iterations are disregarded. This is done to not miscalculate the global cost error which relies on all agents for
    its calculation.
    """
    iter_reached_by_all = min(len(a.x_per_iteration) for a in agents)
    return [(i, utils.global_cost_error_at_iter(agents, problem, i)) for i in range(iter_reached_by_all)]


def global_gradient_optimality_per_iteration(agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[tuple[X, Y]]:
    r"""
    Calculate the global gradient optimality (y-axis) for each iteration (x-axis).

    Global gradient optimality is defined at
    :func:`~decent_bench.library.core.metrics.metric_utils.global_gradient_optimality_at_iter`.

    All iterations up to and including the last one reached by all *agents* are taken into account, subsequent
    iterations are disregarded. This avoids the curve volatility that occurs when fewer and fewer agents are included in
    the calculation.
    """
    iter_reached_by_all = min(len(a.x_per_iteration) for a in agents)
    return [(i, utils.global_gradient_optimality_at_iter(agents, i)) for i in range(iter_reached_by_all)]


@dataclass(eq=False)
class PlotMetric:
    """
    Metric for plotting.

    Args:
        x_label: label for the x-axis
        y_label: label for the y-axis
        x_log: log scale applied to the x-axis if true
        y_log: log scale applied to the y-axis if true
        get_data_from_trial: function that takes trial data as input and extracts data to be plotted

    """

    x_label: str
    y_label: str
    x_log: bool
    y_log: bool
    get_data_from_trial: Callable[[list[AgentMetricsView], BenchmarkProblem], Sequence[tuple[X, Y]]]


X = float
Y = float

DEFAULT_PLOT_METRICS = [
    PlotMetric(
        x_label="iteration",
        y_label="global cost error",
        x_log=False,
        y_log=True,
        get_data_from_trial=global_cost_error_per_iteration,
    ),
    PlotMetric(
        x_label="iteration",
        y_label="global gradient optimality",
        x_log=False,
        y_log=True,
        get_data_from_trial=global_gradient_optimality_per_iteration,
    ),
]
"""
- Global cost error (y-axis) per iteration (x-axis). Semi-log plot.
  Details: :func:`global_cost_error_per_iteration`.
- Global gradient optimality (y-axis) per iteration (x-axis). Semi-log plot.
  Details: :func:`global_gradient_optimality_per_iteration`.

:meta hide-value:
"""
