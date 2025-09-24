from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

import decent_bench.metrics.metric_utils as utils
from decent_bench.agent import AgentMetricsView
from decent_bench.benchmark_problem import BenchmarkProblem


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
Assert that only one value exists and return it using :func:`~decent_bench.metrics.metric_utils.single`.
"""
