import math
import random
import warnings
from collections import defaultdict
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes as SubPlot

from decent_bench.library.core.agent import AgentMetricsView
from decent_bench.library.core.benchmark_problem.benchmark_problems import BenchmarkProblem
from decent_bench.library.core.dst_algorithms import DstAlgorithm
from decent_bench.library.core.metrics.plot_metrics.plot_metrics_constructs import PlotMetric, X, Y
from decent_bench.library.core.network import Network
from decent_bench.library.utils.logger import LOGGER

DOC_LINK = "https://decent-bench.readthedocs.io/en/latest/decent_bench.library.core.metrics.plot_metrics.html"
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
MARKERS = ["o", "s", "v", "^", "*", "D", "H", "<", ">", "p"]


def plot(
    resulting_nw_states_per_alg: dict[DstAlgorithm, list[Network]],
    problem: BenchmarkProblem,
    metrics: list[PlotMetric],
) -> None:
    """
    Plot the execution results with one subplot per metric.

    Each algorithm's curve is its mean across the trials.

    Args:
        resulting_nw_states_per_alg: resulting network states from the trial executions, grouped by algorithm
        problem: benchmark problem whose properties, e.g.
            :attr:`~decent_bench.library.core.benchmark_problem.benchmark_problems.BenchmarkProblem.optimal_x`,
            are used for metric calculations
        metrics: metrics to calculate and plot

    Raises:
        RuntimeError: if the current figure's manager can't be retrieved

    """
    if not metrics:
        return
    metric_subplots: list[tuple[PlotMetric, SubPlot]] = _create_metric_subplots(metrics)
    for metric, subplot in metric_subplots:
        for i, (alg, nw_states) in enumerate(resulting_nw_states_per_alg.items()):
            color = COLORS[i] if i < len(COLORS) else [random.random() for _ in range(3)]
            marker = MARKERS[i] if i < len(MARKERS) else random.choice(MARKERS)
            data_per_trial: list[Sequence[tuple[X, Y]]] = _get_data_per_trial(nw_states, problem, metric)
            flattened_data: list[tuple[X, Y]] = [d for trial in data_per_trial for d in trial]
            if not np.isfinite(flattened_data).all():
                msg = f"Skipping plot {metric.y_label}/{metric.x_label} for {alg.name}: found nan or inf in datapoints."
                LOGGER.warning(msg)
                continue
            mean_curve: Sequence[tuple[X, Y]] = _calculate_mean_curve(data_per_trial)
            x, y_mean = zip(*mean_curve, strict=True)
            subplot.plot(x, y_mean, label=alg.name, color=color, marker=marker, linewidth=3, markevery=100)
            y_min, y_max = _calculate_envelope(data_per_trial)
            subplot.fill_between(x, y_min, y_max, color=color, alpha=0.3)
    manager = plt.get_current_fig_manager()
    if not manager:
        raise RuntimeError("Something went wrong, did not receive a FigureManager...")
    manager.full_screen_toggle()
    plt.tight_layout()
    LOGGER.info(f"Metric definitions can be found here: {DOC_LINK}")
    plt.show()


def _create_metric_subplots(metrics: list[PlotMetric]) -> list[tuple[PlotMetric, SubPlot]]:
    subplots_per_row = 2
    n_metrics = len(metrics)
    n_rows = math.ceil(n_metrics / subplots_per_row)
    fig, subplots = plt.subplots(nrows=n_rows, ncols=subplots_per_row)
    subplots = subplots.flatten()
    for sp in subplots[n_metrics:]:
        fig.delaxes(sp)
    for sp in subplots:
        sp.legend()
    metric_subplots = list(zip(metrics, subplots, strict=True))
    for metric, sp in metric_subplots:
        sp.set_xlabel(metric.x_label)
        sp.set_ylabel(metric.y_label)
        if metric.x_log:
            sp.set_xscale("log")
        if metric.y_log:
            sp.set_yscale("log")
    return metric_subplots


def _get_data_per_trial(
    resulting_nw_states: list[Network], problem: BenchmarkProblem, metric: PlotMetric
) -> list[Sequence[tuple[X, Y]]]:
    data_per_trial: list[Sequence[tuple[X, Y]]] = []
    for nw in resulting_nw_states:
        agent_metrics_views = [AgentMetricsView.from_agent(a) for a in nw.get_all_agents()]
        with warnings.catch_warnings(action="ignore"):
            trial_data = metric.get_data_from_trial(agent_metrics_views, problem)
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
