from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from decent_bench.agent import AgentMetricsView
from decent_bench.benchmark_problem import BenchmarkProblem


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

    get_data_from_trial: Callable[[list[AgentMetricsView], BenchmarkProblem], Sequence[tuple[X, Y]]]
    x_label: str
    y_label: str
    x_log: bool = False
    y_log: bool = True


X = float
Y = float
