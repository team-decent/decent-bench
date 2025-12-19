import math
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes as SubPlot

import decent_bench.metrics.metric_utils as utils
from decent_bench.agents import AgentMetricsView
from decent_bench.benchmark_problem import BenchmarkProblem
from decent_bench.distributed_algorithms import Algorithm
from decent_bench.utils.logger import LOGGER

X = float
Y = float


@dataclass
class ComputationalCost:
    """Computational costs associated with an algorithm for plot metrics."""

    function: float = 1.0
    gradient: float = 1.0
    hessian: float = 1.0
    proximal: float = 1.0
    communication: float = 1.0


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


class RegretPerIteration(PlotMetric):
    r"""
    Global regret (y-axis) per iteration (x-axis).

    Global regret is defined as:

    .. include:: snippets/global_cost_error.rst

    All iterations up to and including the last one reached by all *agents* are taken into account, subsequent
    iterations are disregarded. This is done to not miscalculate the global cost error which relies on all agents for
    its calculation.
    """

    x_label: str = "iteration"
    y_label: str = "regret"

    def get_data_from_trial(self, agents: list[AgentMetricsView], problem: BenchmarkProblem) -> list[tuple[X, Y]]:  # noqa: D102
        # Determine the set of recorded iterations common to all agents and use those
        return [(i, utils.regret(agents, problem, i)) for i in utils.common_sorted_iterations(agents)]


class GradientNormPerIteration(PlotMetric):
    r"""
    Global gradient norm (y-axis) per iteration (x-axis).

    Global gradient norm is defined as:

    .. include:: snippets/global_gradient_optimality.rst

    All iterations up to and including the last one reached by all *agents* are taken into account, subsequent
    iterations are disregarded. This avoids the curve volatility that occurs when fewer and fewer agents are
    included in the calculation.
    """

    x_label: str = "iteration"
    y_label: str = "gradient norm"

    def get_data_from_trial(self, agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[tuple[X, Y]]:  # noqa: D102
        # Determine the set of recorded iterations common to all agents and use those
        return [(i, utils.gradient_norm(agents, i)) for i in utils.common_sorted_iterations(agents)]


DEFAULT_PLOT_METRICS = [
    RegretPerIteration(x_log=False, y_log=True),
    GradientNormPerIteration(x_log=False, y_log=True),
]
"""
- :class:`RegretPerIteration` (semi-log)
- :class:`GradientNormPerIteration` (semi-log)

:meta hide-value:
"""

PLOT_METRICS_DOC_LINK = "https://decent-bench.readthedocs.io/en/latest/api/decent_bench.metrics.plot_metrics.html"
COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#34495e",
    "#16a085",
    "#686901",
]
MARKERS = ["o", "s", "v", "^", "*", "D", "H", "<", ">", "p", "P", "X"]
STYLES = ["-", ":", "--", "-.", (5, (10, 3)), (0, (5, 10)), (0, (3, 1, 1, 1))]


def plot(  # noqa: PLR0914
    resulting_agent_states: dict[Algorithm, list[list[AgentMetricsView]]],
    problem: BenchmarkProblem,
    metrics: list[PlotMetric],
    computational_cost: ComputationalCost | None,
    x_axis_scaling: float = 1e-4,
) -> None:
    """
    Plot the execution results with one subplot per metric.

    Each algorithm's curve is its mean across the trials. The surrounding envelope is the min and max across the trials.

    Args:
        resulting_agent_states: resulting agent states from the trial executions, grouped by algorithm
        problem: benchmark problem whose properties, e.g.
            :attr:`~decent_bench.benchmark_problem.BenchmarkProblem.x_optimal`, are used for metric calculations
        metrics: metrics to calculate and plot
        computational_cost: computational cost settings for plot metrics, if ``None`` x-axis will be iterations instead
            of computational cost
        x_axis_scaling: scaling factor for computational cost x-axis, used to convert the cost units into more
            manageable units for plotting. Only used if ``computational_cost`` is provided.

    Note:
        Computational cost can be interpreted as the cost of running the algorithm on a specific hardware setup.
        Therefore the computational cost could be seen as the number of operations performed (similar to FLOPS) but
        weighted by the time or energy it takes to perform them on the specific hardware.

        .. include:: snippets/computational_cost.rst

    Raises:
        RuntimeError: if the figure manager can't be retrieved

    """
    if not metrics:
        return
    LOGGER.info(f"Plot metric definitions can be found here: {PLOT_METRICS_DOC_LINK}")
    use_cost = computational_cost is not None
    metric_subplots: list[tuple[PlotMetric, SubPlot]] = _create_metric_subplots(metrics, use_cost)
    with utils.MetricProgressBar() as progress:
        plot_task = progress.add_task(
            "Generating plots", total=len(metric_subplots) * len(resulting_agent_states), status=""
        )
        for metric, subplot in metric_subplots:
            progress.update(
                plot_task,
                status=f"Task: {metric.y_label} vs {_get_formatted_x_label(metric.x_label, use_cost)}",
            )
            for i, (alg, agent_states) in enumerate(resulting_agent_states.items()):
                marker, linestyle, color = _get_marker_style_color(i)
                data_per_trial: list[Sequence[tuple[X, Y]]] = _get_data_per_trial(agent_states, problem, metric)
                flattened_data: list[tuple[X, Y]] = [d for trial in data_per_trial for d in trial]
                if not np.isfinite(flattened_data).all():
                    msg = (
                        f"Skipping plot {metric.y_label}/{metric.x_label} for {alg.name}: "
                        "found nan or inf in datapoints."
                    )
                    LOGGER.warning(msg)
                    continue
                mean_curve: Sequence[tuple[X, Y]] = _calculate_mean_curve(data_per_trial)
                x, y_mean = zip(*mean_curve, strict=True)
                if computational_cost is not None:
                    total_computational_cost = _calc_total_cost(agent_states, computational_cost)
                    x = tuple(val * total_computational_cost * x_axis_scaling for val in x)
                subplot.plot(
                    x,
                    y_mean,
                    label=alg.name,
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                    markevery=max(1, int(len(x) / 20)),
                )
                y_min, y_max = _calculate_envelope(data_per_trial)
                subplot.fill_between(x, y_min, y_max, color=color, alpha=0.1)
                progress.advance(plot_task)
            subplot.legend()
        progress.update(plot_task, status="Finalizing plots")
    manager = plt.get_current_fig_manager()
    if not manager:
        raise RuntimeError("Something went wrong, did not receive a FigureManager...")
    plt.tight_layout(pad=1.2)
    plt.show()


def _create_metric_subplots(metrics: list[PlotMetric], use_cost: bool) -> list[tuple[PlotMetric, SubPlot]]:
    subplots_per_row = 2
    n_metrics = len(metrics)
    n_rows = math.ceil(n_metrics / subplots_per_row)
    fig, subplots = plt.subplots(nrows=n_rows, ncols=subplots_per_row)
    subplots = subplots.flatten()
    for sp in subplots[n_metrics:]:
        fig.delaxes(sp)
    metric_subplots = list(zip(metrics, subplots[:n_metrics], strict=True))
    for metric, sp in metric_subplots:
        sp.set_xlabel(_get_formatted_x_label(metric.x_label, use_cost))
        sp.set_ylabel(metric.y_label)
        if metric.x_log:
            sp.set_xscale("log")
        if metric.y_log:
            sp.set_yscale("log")
    return metric_subplots


def _get_marker_style_color(
    index: int,
) -> tuple[str, Sequence[int | tuple[int, int, int, int] | str | tuple[int, int]], str]:
    """
    Get deterministic unique marker, line style, and color for a given index.

    Cycles through all combinations to ensure the first n indices (where n =
    len(MARKERS) * len(STYLES)) are unique. Colors cycle based on index,
    markers cycle first, then styles to maximize marker distinctiveness for B&W printing.
    """
    # Calculate total unique combinations
    n_combinations = len(MARKERS) * len(STYLES)

    # Reduce index to valid range
    idx = index % n_combinations

    color_idx = index % len(COLORS)
    marker_idx = idx % len(MARKERS)
    style_idx = (idx // len(MARKERS)) % len(STYLES)

    return MARKERS[marker_idx], STYLES[style_idx], COLORS[color_idx]


def _get_formatted_x_label(x_label: str, use_cost: bool) -> str:
    return f"{x_label} (computational cost units)" if use_cost else x_label


def _calc_total_cost(agent_states: list[list[AgentMetricsView]], computational_cost: ComputationalCost) -> float:
    mean_function_calls = np.mean([a.n_function_calls for agents in agent_states for a in agents])
    mean_gradient_calls = np.mean([a.n_gradient_calls for agents in agent_states for a in agents])
    mean_hessian_calls = np.mean([a.n_hessian_calls for agents in agent_states for a in agents])
    mean_proximal_calls = np.mean([a.n_proximal_calls for agents in agent_states for a in agents])
    mean_communication_calls = np.mean([a.n_sent_messages for agents in agent_states for a in agents])

    return float(
        computational_cost.function * mean_function_calls
        + computational_cost.gradient * mean_gradient_calls
        + computational_cost.hessian * mean_hessian_calls
        + computational_cost.proximal * mean_proximal_calls
        + computational_cost.communication * mean_communication_calls
    )


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
