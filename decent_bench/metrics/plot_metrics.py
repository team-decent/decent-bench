import math
import random
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes as SubPlot

import decent_bench.metrics.metric_utils as utils
from decent_bench.agent import AgentMetricsView
from decent_bench.benchmark_problem import BenchmarkProblem
from decent_bench.distributed_algorithms import DstAlgorithm
from decent_bench.utils.logger import LOGGER

X = float
Y = float


class PlotMetric(ABC):
    """
    Metric to plot at the end of the benchmarking execution.

    Args:
        x_log: whether to apply log scaling to the x-axis.
        y_log: whether to apply log scaling to the y-axis.

    """

    def __init__(self, *, x_log: bool = False, y_log: bool = True):
        self.x_log = x_log
        self.y_log = y_log

    @property
    @abstractmethod
    def x_label(self) -> str:
        """Label for the x-axis."""

    @property
    @abstractmethod
    def y_label(self) -> str:
        """Label for the y-axis."""

    @abstractmethod
    def get_data_from_trial(self, agents: list[AgentMetricsView], problem: BenchmarkProblem) -> Sequence[tuple[X, Y]]:
        """Extract trial data in the form of (x, y) datapoints."""


class GlobalCostErrorPerIteration(PlotMetric):
    r"""
    Global cost error (y-axis) per iteration (x-axis).

    Global cost error is defined as:

    .. include:: snippets/global_cost_error.rst

    All iterations up to and including the last one reached by all *agents* are taken into account, subsequent
    iterations are disregarded. This is done to not miscalculate the global cost error which relies on all agents for
    its calculation.
    """

    x_label: str = "iteration"
    y_label: str = "global cost error"

    def get_data_from_trial(self, agents: list[AgentMetricsView], problem: BenchmarkProblem) -> list[tuple[X, Y]]:  # noqa: D102
        iter_reached_by_all = min(len(a.x_per_iteration) for a in agents)
        return [(i, utils.global_cost_error_at_iter(agents, problem, i)) for i in range(iter_reached_by_all)]


class GlobalGradientOptimalityPerIteration(PlotMetric):
    r"""
    Global gradient optimality (y-axis) per iteration (x-axis).

    Global gradient optimality is defined as:

    .. include:: snippets/global_gradient_optimality.rst

    All iterations up to and including the last one reached by all *agents* are taken into account, subsequent
    iterations are disregarded. This avoids the curve volatility that occurs when fewer and fewer agents are
    included in the calculation.
    """

    x_label: str = "iteration"
    y_label: str = "global gradient optimality"

    def get_data_from_trial(self, agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[tuple[X, Y]]:  # noqa: D102
        iter_reached_by_all = min(len(a.x_per_iteration) for a in agents)
        return [(i, utils.global_gradient_optimality_at_iter(agents, i)) for i in range(iter_reached_by_all)]


DEFAULT_PLOT_METRICS = [
    GlobalCostErrorPerIteration(x_log=False, y_log=True),
    GlobalGradientOptimalityPerIteration(x_log=False, y_log=True),
]
"""
- :class:`GlobalCostErrorPerIteration` (semi-log)
- :class:`GlobalGradientOptimalityPerIteration` (semi-log)

:meta hide-value:
"""

PLOT_METRICS_DOC_LINK = "https://decent-bench.readthedocs.io/en/latest/api/decent_bench.metrics.plot_metrics.html"
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
MARKERS = ["o", "s", "v", "^", "*", "D", "H", "<", ">", "p"]


def plot(
    resulting_agent_states: dict[DstAlgorithm, list[list[AgentMetricsView]]],
    problem: BenchmarkProblem,
    metrics: list[PlotMetric],
) -> None:
    """
    Plot the execution results with one subplot per metric.

    Each algorithm's curve is its mean across the trials. The surrounding envelope is the min and max across the trials.

    Args:
        resulting_agent_states: resulting agent states from the trial executions, grouped by algorithm
        problem: benchmark problem whose properties, e.g.
            :attr:`~decent_bench.benchmark_problem.BenchmarkProblem.optimal_x`, are used for metric calculations
        metrics: metrics to calculate and plot

    Raises:
        RuntimeError: if the figure manager can't be retrieved

    """
    if not metrics:
        return
    LOGGER.info(f"Plot metric definitions can be found here: {PLOT_METRICS_DOC_LINK}")
    metric_subplots: list[tuple[PlotMetric, SubPlot]] = _create_metric_subplots(metrics)
    for metric, subplot in metric_subplots:
        for i, (alg, agent_states) in enumerate(resulting_agent_states.items()):
            color = COLORS[i] if i < len(COLORS) else [random.random() for _ in range(3)]
            marker = MARKERS[i] if i < len(MARKERS) else random.choice(MARKERS)
            data_per_trial: list[Sequence[tuple[X, Y]]] = _get_data_per_trial(agent_states, problem, metric)
            flattened_data: list[tuple[X, Y]] = [d for trial in data_per_trial for d in trial]
            if not np.isfinite(flattened_data).all():
                msg = f"Skipping plot {metric.y_label}/{metric.x_label} for {alg.name}: found nan or inf in datapoints."
                LOGGER.warning(msg)
                continue
            mean_curve: Sequence[tuple[X, Y]] = _calculate_mean_curve(data_per_trial)
            x, y_mean = zip(*mean_curve, strict=True)
            subplot.plot(x, y_mean, label=alg.name, color=color, marker=marker, markevery=max(1, int(len(x) / 20)))
            y_min, y_max = _calculate_envelope(data_per_trial)
            subplot.fill_between(x, y_min, y_max, color=color, alpha=0.1)
        subplot.legend()
    manager = plt.get_current_fig_manager()
    if not manager:
        raise RuntimeError("Something went wrong, did not receive a FigureManager...")
    manager.full_screen_toggle()
    plt.tight_layout()
    plt.show()


def _create_metric_subplots(metrics: list[PlotMetric]) -> list[tuple[PlotMetric, SubPlot]]:
    subplots_per_row = 2
    n_metrics = len(metrics)
    n_rows = math.ceil(n_metrics / subplots_per_row)
    fig, subplots = plt.subplots(nrows=n_rows, ncols=subplots_per_row)
    subplots = subplots.flatten()
    for sp in subplots[n_metrics:]:
        fig.delaxes(sp)
    metric_subplots = list(zip(metrics, subplots[:n_metrics], strict=True))
    for metric, sp in metric_subplots:
        sp.set_xlabel(metric.x_label)
        sp.set_ylabel(metric.y_label)
        if metric.x_log:
            sp.set_xscale("log")
        if metric.y_log:
            sp.set_yscale("log")
    return metric_subplots


def _get_data_per_trial(
    agents_per_trial: list[list[AgentMetricsView]], problem: BenchmarkProblem, metric: PlotMetric
) -> list[Sequence[tuple[X, Y]]]:
    data_per_trial: list[Sequence[tuple[X, Y]]] = []
    for agents in agents_per_trial:
        with warnings.catch_warnings(action="ignore"):
            trial_data = metric.get_data_from_trial(agents, problem)
        data_per_trial.append(trial_data)
    return data_per_trial


def _calculate_mean_curve(data_per_trial: list[Sequence[tuple[X, Y]]]) -> list[tuple[X, Y]]:
    all_y_per_x: dict[X, list[Y]] = defaultdict(list)
    for trial_data in data_per_trial:
        for x, y in trial_data:
            all_y_per_x[x].append(y)
    return [(x, np.mean(y_li, dtype=float)) for x, y_li in all_y_per_x.items()]


def _calculate_envelope(data_per_trial: list[Sequence[tuple[X, Y]]]) -> tuple[list[Y], list[Y]]:
    y_span_per_x: dict[X, dict[str, Y]] = defaultdict(lambda: {"y_min": np.inf, "y_max": -np.inf})
    for trial_data in data_per_trial:
        for x, y in trial_data:
            y_span_per_x[x]["y_min"] = min(y_span_per_x[x]["y_min"], y)
            y_span_per_x[x]["y_max"] = max(y_span_per_x[x]["y_max"], y)
    return [v["y_min"] for v in y_span_per_x.values()], [v["y_max"] for v in y_span_per_x.values()]
