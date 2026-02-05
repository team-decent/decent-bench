import math
import pathlib
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import tabulate as tb
from matplotlib.axes import Axes as SubPlot
from matplotlib.figure import Figure
from scipy import stats

import decent_bench.metrics.metric_utils as utils
from decent_bench.agents import AgentMetricsView
from decent_bench.benchmark_problem import BenchmarkProblem
from decent_bench.distributed_algorithms import Algorithm
from decent_bench.utils.logger import LOGGER

Statistic = Callable[[Sequence[float]], float]
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


class Metric(ABC):
    """
    Abstract base class for metrics.

    Args:
        statistics: sequence of statistics such as :func:`min`, :func:`sum`, and :func:`~numpy.average` used for
            aggregating the data retrieved with :func:`get_data_from_trial` into a single value, each statistic gets its
            own row in the table
        x_log: whether to apply log scaling to the x-axis in plots.
        y_log: whether to apply log scaling to the y-axis in plots.
        fmt: format string used to format the values in the table, defaults to ".2e". Common formats include:
            - ".2e": scientific notation with 2 decimal places
            - ".3f": fixed-point notation with 3 decimal places
            - ".4g": general format with 4 significant digits
            - ".1%": percentage format with 1 decimal place

            Where the integer specifies the precision.
            See :meth:`str.format` documentation for details on the format string options.

    """

    def __init__(
        self,
        statistics: Sequence[Statistic],
        *,
        x_log: bool = False,
        y_log: bool = True,
        fmt: str = ".2e",
    ) -> None:
        self.statistics = statistics
        self.x_log = x_log
        self.y_log = y_log
        self.fmt = fmt

    @property
    @abstractmethod
    def plot_description(self) -> str:
        """Label for the y-axis in plots."""

    @property
    @abstractmethod
    def table_description(self) -> str:
        """Metric description to display in the table."""

    @property
    def can_diverge(self) -> bool:
        """
        Indicates whether the metric can diverge, i.e. take on infinite or NaN values.

        If True then the table will try to indicate if the has metric diverged.
        Has no real impact on calulations of the metric, will not effect plots.
        """
        return True

    @abstractmethod
    def get_data_from_trial(
        self,
        agents: Sequence[AgentMetricsView],
        problem: BenchmarkProblem,
        iteration: int,
    ) -> Sequence[float]:
        """
        Get the data for this metric from a trial.

        Args:
            agents: the agents being evaluated
            problem: the benchmark problem being evaluated
            iteration: the iteration at which to evaluate the metric, or -1 to use the agents' final x

        Returns:
            a list of floats, one for each agent

        """

    def get_plot_data(self, agents: Sequence[AgentMetricsView], problem: BenchmarkProblem) -> Sequence[tuple[X, Y]]:
        """
        Extract trial data to be used in plots for this metric.

        This is used by :func:`create_plots` to generate plots for this metric.
        By default, it calculates statistics on the intersection of all the iterations
        reached by all agents, but it can be overridden to perform additional
        processing on the data before it is used in plots.
        """
        return [
            (i, float(np.mean(self.get_data_from_trial(agents, problem, i))))
            for i in utils.common_sorted_iterations(agents)
        ]

    def get_table_data(self, agents: Sequence[AgentMetricsView], problem: BenchmarkProblem) -> Sequence[float]:
        """
        Extract trial data to be used in the table for this metric.

        This is used by :func:`create_table` to generate the table for this metric.
        By default, it returns the metric from the last iteration,
        but it can be overridden to perform additional processing on the data before it is used in the table.
        """
        return self.get_data_from_trial(agents, problem, -1)


METRICS_DOC_LINK = "https://decent-bench.readthedocs.io/en/latest/api/decent_bench.metrics.metric_collection.html"
X_LABELS = {
    "iterations": "iterations",
    "computational_cost": "time (computational cost units)",
}
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


def create_plots(  # noqa: PLR0917
    resulting_agent_states: dict[Algorithm, list[list[AgentMetricsView]]],
    problem: BenchmarkProblem,
    metrics: list[Metric],
    computational_cost: ComputationalCost | None,
    x_axis_scaling: float = 1e-4,
    compare_iterations_and_computational_cost: bool = False,
    plot_path: str | None = None,
    plot_grid: bool = True,
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
            manageable units for plotting. Only used if ``computational_cost`` is provided
        compare_iterations_and_computational_cost: whether to plot both metric vs computational cost and
            metric vs iterations. Only used if ``computational_cost`` is provided
        plot_path: optional file path to save the generated plot as an image file (e.g., "plots.png"). If ``None``,
            the plot will only be displayed
        plot_grid: whether to show grid lines on the plots

    Note:
        Computational cost can be interpreted as the cost of running the algorithm on a specific hardware setup.
        Therefore the computational cost could be seen as the number of operations performed (similar to FLOPS) but
        weighted by the time or energy it takes to perform them on the specific hardware.

        .. include:: snippets/computational_cost.rst

    """
    if not metrics:
        return
    LOGGER.info(f"Plot metric definitions can be found here: {METRICS_DOC_LINK}")

    if len(metrics) > 4:
        LOGGER.warning(
            f"Plotting {len(metrics)} (> 4) metrics may result in a cluttered figure. "
            "Consider reducing the number of metrics for better readability."
        )

    did_plot = False
    use_cost = computational_cost is not None
    two_columns = use_cost and compare_iterations_and_computational_cost
    fig, metric_subplots = _create_metric_subplots(
        metrics,
        use_cost,
        compare_iterations_and_computational_cost,
        plot_grid,
    )
    with utils.MetricProgressBar() as progress:
        plot_task = progress.add_task(
            "Generating plots",
            total=len(metric_subplots) * len(resulting_agent_states),
            status="",
        )
        x_label = X_LABELS["computational_cost" if use_cost else "iterations"]
        for metric_index in range(len(metrics)):
            progress.update(
                plot_task,
                status=f"Task: {metrics[metric_index].plot_description} vs {x_label}",
            )
            for i, (alg, agent_states) in enumerate(resulting_agent_states.items()):
                data_per_trial: list[Sequence[tuple[X, Y]]] = _plot_data_per_trial(
                    agent_states,
                    problem,
                    metrics[metric_index],
                )
                if not _is_finite(data_per_trial):
                    msg = (
                        f"Skipping plot {metrics[metric_index].plot_description}/{x_label} "
                        f"for {alg.name}: found nan or inf in datapoints."
                    )
                    LOGGER.warning(msg)
                    progress.advance(plot_task, 2 if two_columns else 1)
                    continue
                _plot(
                    metric_subplots,
                    data_per_trial,
                    computational_cost,
                    compare_iterations_and_computational_cost,
                    x_axis_scaling,
                    agent_states,
                    alg,
                    metric_index,
                    i,
                )
                did_plot = True
                progress.advance(plot_task, 2 if two_columns else 1)
        progress.update(plot_task, status="Finalizing plots")

    if not did_plot:
        LOGGER.warning("No plots were generated due to invalid data.")
        return

    _show_figure(fig, metric_subplots, two_columns, plot_path)


def _create_metric_subplots(
    metrics: list[Metric],
    use_cost: bool,
    compare_iterations_and_computational_cost: bool,
    plot_grid: bool,
) -> tuple[Figure, list[SubPlot]]:
    n_cols = 2 if use_cost and compare_iterations_and_computational_cost else 1
    n_plots = len(metrics) * n_cols
    n_rows = math.ceil(n_plots / n_cols)

    fig, subplot_axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        sharex="col",
        sharey="row",
        layout="constrained",
    )
    if isinstance(subplot_axes, SubPlot):
        subplots: list[SubPlot] = [subplot_axes]
    else:
        subplots = subplot_axes.flatten()

    if subplots is None:
        raise RuntimeError("Something went wrong, did not receive subplot axes...")

    for sp in subplots[n_plots + n_cols :]:
        fig.delaxes(sp)

    for i in range(n_plots):
        metric = metrics[i // (2 if n_cols == 2 else 1)]
        sp = subplots[i]

        # Only set x label for subplots in the last row
        if i // n_cols == n_rows - 1:
            # For comparison mode, right column shows iterations, left shows cost
            if n_cols == 2:
                sp.set_xlabel(X_LABELS["iterations"] if i % 2 == 1 else X_LABELS["computational_cost"])
            else:
                # Single column mode: show cost if enabled, otherwise iterations
                sp.set_xlabel(X_LABELS["computational_cost" if use_cost else "iterations"])

        # Only set y label for left column subplots
        if i % n_cols == 0:
            sp.set_ylabel(metric.plot_description)

        if metric.x_log:
            sp.set_xscale("log")
        if metric.y_log:
            sp.set_yscale("log")

        if plot_grid:
            sp.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)  # noqa: FBT003

    return fig, subplots[:n_plots]


def _show_figure(
    fig: Figure,
    metric_subplots: list[SubPlot],
    two_columns: bool,
    plot_path: str | None = None,
) -> None:
    manager = plt.get_current_fig_manager()
    if not manager:
        raise RuntimeError("Something went wrong, did not receive a FigureManager...")

    # Create a single legend at the top of the figure
    handles, labels = metric_subplots[0].get_legend_handles_labels()
    label_cols = min(len(labels), 4 if two_columns else 3)

    # Create the legend to get the height of the legend box
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=label_cols,
        frameon=True,
    )

    if plot_path is not None:
        fig.savefig(plot_path, dpi=300)
        LOGGER.info(f"Saved plot to: {plot_path}")

    plt.show()


def _is_finite(data_per_trial: list[Sequence[tuple[X, Y]]]) -> bool:
    flattened_data: list[tuple[X, Y]] = [d for trial in data_per_trial for d in trial]
    return np.isfinite(flattened_data).all().item()


def _plot(  # noqa: PLR0917
    metric_subplots: list[SubPlot],
    data_per_trial: list[Sequence[tuple[X, Y]]],
    computational_cost: ComputationalCost | None,
    compare_iterations_and_computational_cost: bool,
    x_axis_scaling: float,
    agent_states: list[list[AgentMetricsView]],
    alg: Algorithm,
    metric_index: int,
    iteration: int,
) -> None:
    use_cost = computational_cost is not None
    subplot_idx = metric_index * (2 if use_cost and compare_iterations_and_computational_cost else 1)

    mean_curve: Sequence[tuple[X, Y]] = _calculate_mean_curve(data_per_trial)
    x, y_mean = zip(*mean_curve, strict=True)
    y_min, y_max = _calculate_envelope(data_per_trial)
    if computational_cost is not None:
        total_computational_cost = _calc_total_cost(agent_states, computational_cost)
        x_computational = tuple(val * total_computational_cost * x_axis_scaling for val in x)
        if compare_iterations_and_computational_cost:
            # Plot value vs iterations subplot first
            iter_idx = metric_index * 2 + 1
            _plot_subplot(metric_subplots[iter_idx], x, y_mean, y_min, y_max, alg.name, iteration)
        x = x_computational
    _plot_subplot(metric_subplots[subplot_idx], x, y_mean, y_min, y_max, alg.name, iteration)


def _plot_subplot(  # noqa: PLR0917
    subplot: SubPlot,
    x: Sequence[float],
    y_mean: Sequence[float],
    y_min: Sequence[float],
    y_max: Sequence[float],
    label: str,
    iteration: int,
) -> None:
    marker, linestyle, color = _get_marker_style_color(iteration)
    subplot.plot(
        x,
        y_mean,
        label=label,
        color=color,
        marker=marker,
        linestyle=linestyle,
        markevery=max(1, int(len(x) / 10)),
    )
    subplot.fill_between(x, y_min, y_max, color=color, alpha=0.1)


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


def _plot_data_per_trial(
    agents_per_trial: list[list[AgentMetricsView]],
    problem: BenchmarkProblem,
    metric: Metric,
) -> list[Sequence[tuple[X, Y]]]:
    data_per_trial: list[Sequence[tuple[X, Y]]] = []
    for agents in agents_per_trial:
        with warnings.catch_warnings(action="ignore"):
            trial_data = metric.get_plot_data(agents, problem)
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


def create_table(
    resulting_agent_states: dict[Algorithm, list[list[AgentMetricsView]]],
    problem: BenchmarkProblem,
    metrics: list[Metric],
    confidence_level: float,
    table_fmt: Literal["grid", "latex"],
    *,
    table_path: str | None = None,
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
        table_path: optional path to save the table as a text file, if not provided the table is not saved to a file

    """
    if not metrics:
        return
    LOGGER.info(f"Table metric definitions can be found here: {METRICS_DOC_LINK}")
    algs = list(resulting_agent_states)
    headers = ["Metric (statistic)"] + [alg.name for alg in algs]
    rows: list[list[str]] = []
    statistics_abbr = {"average": "avg", "median": "mdn"}
    with warnings.catch_warnings(action="ignore"), utils.MetricProgressBar() as progress:
        n_statistics = sum(len(metric.statistics) for metric in metrics)
        table_task = progress.add_task("Generating table", total=n_statistics, status="")
        for metric in metrics:
            progress.update(table_task, status=f"Task: {metric.table_description}")
            data_per_trial = [_table_data_per_trial(resulting_agent_states[a], problem, metric) for a in algs]
            for statistic in metric.statistics:
                row = [f"{metric.table_description} ({statistics_abbr.get(statistic.__name__) or statistic.__name__})"]
                for i in range(len(algs)):
                    agg_data_per_trial = [statistic(trial) for trial in data_per_trial[i]]
                    mean, margin_of_error = _calculate_mean_and_margin_of_error(agg_data_per_trial, confidence_level)
                    formatted_confidence_interval = _format_confidence_interval(
                        mean,
                        margin_of_error,
                        metric.fmt,
                        metric.can_diverge,
                    )
                    row.append(formatted_confidence_interval)
                rows.append(row)
                progress.advance(table_task)
        progress.update(table_task, status="Finalizing table")
    formatted_table = tb.tabulate(rows, headers, tablefmt=table_fmt)
    LOGGER.info("\n" + formatted_table)
    if table_path:
        pathlib.Path(table_path).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(table_path).write_text(formatted_table, encoding="utf-8")


def _table_data_per_trial(
    agents_per_trial: list[list[AgentMetricsView]],
    problem: BenchmarkProblem,
    metric: Metric,
) -> list[Sequence[float]]:
    data_per_trial: list[Sequence[float]] = []
    for agents in agents_per_trial:
        trial_data = metric.get_table_data(agents, problem)
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


def _format_confidence_interval(mean: float, margin_of_error: float, fmt: str, can_diverge: bool) -> str:
    if not _is_valid_float_format_spec(fmt):
        LOGGER.warning(f"Invalid format string '{fmt}', defaulting to scientific notation")
        fmt = ".2e"

    formatted_confidence_interval = f"{mean:{fmt}} \u00b1 {margin_of_error:{fmt}}"

    if any(np.isnan([mean, margin_of_error])) and can_diverge:
        formatted_confidence_interval += " (diverged?)"

    return formatted_confidence_interval


def _is_valid_float_format_spec(fmt: str) -> bool:
    """
    Validate that the given format spec can be used to format a float.

    This avoids attempting to format real values with an invalid format string.

    """
    try:
        f"{0.01:{fmt}}"
    except (ValueError, TypeError):
        return False
    return True
