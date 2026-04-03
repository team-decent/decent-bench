import math
import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes as SubPlot
from matplotlib.figure import Figure

import decent_bench.metrics.metric_utils as utils
from decent_bench.agents import AgentMetricsView
from decent_bench.distributed_algorithms import Algorithm
from decent_bench.metrics._computational_cost import ComputationalCost
from decent_bench.metrics._metric import Metric, X, Y
from decent_bench.networks import Network
from decent_bench.utils._metric_helpers import _flatten_plot_metrics
from decent_bench.utils.logger import LOGGER

if TYPE_CHECKING:
    from decent_bench.benchmark import BenchmarkProblem, MetricResult


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
# Upper bound used when deciding whether a point is still plottable on log y-axis.
# Values above this threshold are treated as divergence for plotting and trigger truncation.
MAX_LOG_PLOT_VALUE = 1e100
# thresholds to decide when to plot the legend as a separate figure
LEGEND_MAX_COLS = 3
SEPARATE_LEGEND_MIN_LABELS = 9
SEPARATE_LEGEND_MIN_ROWS = 4
SEPARATE_LEGEND_MAX_LABEL_LENGTH = 24


def display_plots(
    metrics_result: "MetricResult",
    *,
    computational_cost: ComputationalCost | None,
    scale_x_axis: float = 1e-4,
    compare_iterations_and_computational_cost: bool = False,
    individual_plots: bool = False,
    plot_grid: bool = True,
    plot_format: Literal["png", "pdf", "svg"] = "png",
    plot_path: Path | None = None,
) -> None:
    """
    Display plots for the metric results.

    Each algorithm's curve is its mean across the trials. The surrounding envelope is the min and max across the trials.
    If metrics is a list of lists, each inner list will be plotted in a separate figure. Otherwise groups of 3 metrics
    will be plotted together in subplots of the same figure.

    Args:
        metrics_result: result of metrics computation containing the metrics to plot and the data to plot them with.
        computational_cost: computational cost settings for plot metrics, if ``None`` x-axis will be iterations instead
            of computational cost.
        scale_x_axis: scaling factor for computational cost x-axis, used to convert the cost units into more
            manageable units for plotting. Only used if ``computational_cost`` is provided.
        compare_iterations_and_computational_cost: whether to plot both metric vs computational cost and
            metric vs iterations. Only used if ``computational_cost`` is provided.
        individual_plots: whether to create individual plots for each metric instead of subplots.
        plot_grid: whether to show grid lines on the plots.
        plot_format: format to save plots in, defaults to ``png``. Can be ``png``, ``pdf``, or ``svg``.
        plot_path: optional directory path to save the generated plots as image files.
            Will be saved as "plot.png" or "plot_fig1.png", "plot_fig2.png", etc. if multiple figures.
            If not provided, the plots will only be displayed.

    Note:
        Computational cost can be interpreted as the cost of running the algorithm on a specific hardware setup.
        Therefore the computational cost could be seen as the number of operations performed (similar to FLOPS) but
        weighted by the time or energy it takes to perform them on the specific hardware.

        Plots are generated from precomputed metric trajectories. If a trajectory diverges, only its finite/plottable
        part is shown (see :func:`compute_plots`).

        For log-scale plots, non-positive y values (due to floating point errors) are replaced with a small positive
        value for rendering stability. A warning is logged when this replacement happens.

        .. include:: snippets/computational_cost.rst

    """
    if not metrics_result.plot_results or not metrics_result.plot_metrics:
        LOGGER.warning("No plot metrics to display.")
        return

    plot_results = metrics_result.plot_results
    metrics = metrics_result.plot_metrics
    agent_metrics = metrics_result.agent_metrics

    # Normalize metrics into list of groups (each group will be one figure)
    metric_groups = _organize_metrics_into_groups(metrics, individual_plots)

    use_cost = computational_cost is not None
    two_columns = use_cost and compare_iterations_and_computational_cost

    # Create figures and plot data
    all_figures = _create_and_plot_figures(
        metric_groups,
        plot_results,
        agent_metrics,
        use_cost,
        two_columns,
        computational_cost=computational_cost,
        scale_x_axis=scale_x_axis,
        plot_grid=plot_grid,
    )

    if not all_figures:
        LOGGER.warning("No plots were generated due to invalid data.")
        return

    # Save and show figures
    _save_and_show_figures(all_figures, plot_path=plot_path, plot_format=plot_format)


def compute_plots(  # noqa: PLR0914
    resulting_agent_states: dict[Algorithm[Network], list[list[AgentMetricsView]]],
    problem: "BenchmarkProblem",
    metrics: list[Metric] | list[list[Metric]],
) -> Mapping[
    Algorithm[Network], Mapping[Metric, tuple[Sequence[float], Sequence[float], Sequence[float], Sequence[float]]]
]:
    """
    Compute plot data for metrics.

    Each algorithm's curve is its mean across the trials. The surrounding envelope is the min and max across the trials.

    Args:
        resulting_agent_states: resulting agent states from the trial executions, grouped by algorithm
        problem: benchmark problem whose properties are used for metric calculations
        metrics: metrics to calculate and plot. If a list of lists is provided, each inner list will be plotted in a
            separate figure. Otherwise groups of 3 metrics will be plotted together in subplots of the same figure

    Returns:
        A nested dictionary containing plot data for each algorithm and metric, structured as
        {Algorithm: {Metric: (x, y_mean, y_min, y_max)}}, where x is the sequence of x values for the plot,
        y_mean is the sequence of mean y values across trials for each x, and y_min and y_max are the sequences
        of minimum and maximum y values across trials for each x, respectively.

    Note:
        Plot trajectories are truncated per trial at the first datapoint that is either non-finite or has
        y > ``MAX_LOG_PLOT_VALUE``. Aggregation is then performed over the common prefix of the remaining trials.

        If no plottable prefix remains for an algorithm/metric pair, that pair is omitted from the returned mapping.
        This can lead to a metric showing ``nan`` in tables (final iteration diverged) while still being visible in
        plots (finite prefix exists).

    """
    if not metrics:
        return {}

    flat_metrics = _flatten_plot_metrics(metrics)

    algs = list(resulting_agent_states)
    results: dict[
        Algorithm[Network],
        dict[Metric, tuple[Sequence[float], Sequence[float], Sequence[float], Sequence[float]]],
    ] = {alg: {} for alg in algs}

    with utils.MetricProgressBar() as progress:
        total_plots = len(flat_metrics) * len(resulting_agent_states)
        plot_task = progress.add_task(
            "Computing plot metrics",
            total=total_plots,
            status="",
        )

        for metric in flat_metrics:
            progress.update(plot_task, status=f"Task: {metric.plot_description}")
            available, reason = metric.is_available(problem)

            if not available:
                LOGGER.warning(f"Skipping plot metric '{metric.plot_description}' because it is unavailable: {reason}")
                progress.advance(plot_task, advance=len(resulting_agent_states))
                continue

            for alg, agent_states in resulting_agent_states.items():
                data_per_trial: list[Sequence[tuple[X, Y]]] = _plot_data_per_trial(
                    agent_states,
                    problem,
                    metric,
                )

                truncated_data_per_trial, had_non_finite = _truncate_to_common_finite_prefix(data_per_trial)

                if not truncated_data_per_trial:
                    if had_non_finite:
                        LOGGER.warning(
                            f"Skipping plot computation for {metric.plot_description} and {alg.name}: "
                            "all trials diverged before the first plottable datapoint."
                        )
                    else:
                        LOGGER.warning(
                            f"Skipping plot computation for {metric.plot_description} and {alg.name}: "
                            "metric produced no datapoints."
                        )
                    progress.advance(plot_task)
                    continue

                if had_non_finite:
                    retained_trials = len(truncated_data_per_trial)
                    total_trials = len(data_per_trial)
                    retained_points = len(truncated_data_per_trial[0])
                    LOGGER.info(
                        f"Truncating plot computation for {metric.plot_description} and {alg.name} "
                        "at the first non-finite or over-threshold datapoint; retained "
                        f"{retained_points} point(s) from {retained_trials}/{total_trials} trial(s)."
                    )

                mean_curve: Sequence[tuple[X, Y]] = _calculate_mean_curve(truncated_data_per_trial)
                x, y_mean = zip(*mean_curve, strict=True)
                y_min, y_max = _calculate_envelope(truncated_data_per_trial)

                results[alg][metric] = (x, y_mean, y_min, y_max)
                progress.advance(plot_task)

        progress.update(plot_task, status="Plot computation complete")

    return results


def _create_and_plot_figures(
    metric_groups: list[list[Metric]],
    plot_results: Mapping[
        Algorithm[Network],
        Mapping[Metric, tuple[Sequence[float], Sequence[float], Sequence[float], Sequence[float]]],
    ],
    resulting_agent_states: Mapping[Algorithm[Network], Sequence[Sequence[AgentMetricsView]]] | None,
    use_cost: bool,
    two_columns: bool,
    *,
    computational_cost: ComputationalCost | None,
    scale_x_axis: float,
    plot_grid: bool,
) -> list[tuple[Figure, list[SubPlot]]]:
    """Create figures, plot data, and return non-empty figures."""
    all_figures: list[tuple[Figure, list[SubPlot]]] = []

    # Create a figure for each metric group
    for metric_group in metric_groups:
        fig, metric_subplots = _create_metric_subplots(
            metric_group,
            use_cost,
            two_columns,
            plot_grid,
        )
        all_figures.append((fig, metric_subplots))

    # Now plot all the data
    algs = list(plot_results.keys())

    for group_idx, metric_group in enumerate(metric_groups):
        fig, metric_subplots = all_figures[group_idx]

        for metric_index_in_group, metric in enumerate(metric_group):
            for alg_idx, alg in enumerate(algs):
                # Skip if no data for this metric/algorithm combination
                if metric not in plot_results[alg]:
                    continue

                x, y_mean, y_min, y_max = plot_results[alg][metric]

                if metric.x_log and any(val <= 0 for val in x):
                    offset = 1 - min(x)
                    x = [val + offset for val in x]  # avoid log(a), a<=0 issues
                    LOGGER.warning(
                        f"Metric '{metric.plot_description}' has x_log=True but contains non-positive x values. "
                        f"Added {offset} to all x values for plotting purposes."
                    )
                if metric.y_log and any(val <= 0 for val in y_mean):
                    non_positive_replacement = min(*(val for val in y_mean if val > 0), 1e-8)
                    y_mean = [val if val > 0 else non_positive_replacement for val in y_mean]
                    LOGGER.warning(
                        f"Metric '{metric.plot_description}' has y_log=True but contains non-positive y values for"
                        f"algorithm {alg.name}. These values have been replaced with {non_positive_replacement} for "
                        f"plotting purposes."
                    )
                if metric.y_log and any(val <= 0 for val in y_min):
                    non_positive_replacement = min(*(val for val in y_min if val > 0), 1e-8)
                    y_min = [val if val > 0 else non_positive_replacement for val in y_min]
                if metric.y_log and any(val <= 0 for val in y_max):
                    non_positive_replacement = min(*(val for val in y_max if val > 0), 1e-8)
                    y_max = [val if val > 0 else non_positive_replacement for val in y_max]

                # Determine subplot index
                subplot_idx = metric_index_in_group * (2 if two_columns else 1)

                # Transform x-axis for computational cost if needed
                x_to_plot = x
                if use_cost and computational_cost is not None and resulting_agent_states is not None:
                    agent_states_for_alg = [list(trial) for trial in resulting_agent_states[alg]]
                    total_computational_cost = _calc_total_cost(agent_states_for_alg, computational_cost)
                    x_to_plot = tuple(val * total_computational_cost * scale_x_axis for val in x)

                # Plot iterations version if comparing
                if two_columns:
                    iter_idx = metric_index_in_group * 2 + 1
                    _plot_subplot(metric_subplots[iter_idx], x, y_mean, y_min, y_max, alg.name, alg_idx)

                # Plot main version (cost or iterations)
                _plot_subplot(metric_subplots[subplot_idx], x_to_plot, y_mean, y_min, y_max, alg.name, alg_idx)

    # Filter out empty figures (ones with no data plotted in any subplot)
    return [
        (fig, metric_subplots)
        for fig, metric_subplots in all_figures
        if any(sp.get_legend_handles_labels()[0] for sp in metric_subplots)
    ]


def _save_and_show_figures(
    figures_to_show: list[tuple[Figure, list[SubPlot]]],
    *,
    plot_path: Path | None,
    plot_format: Literal["png", "pdf", "svg"],
) -> None:
    """Add legends, save figures to files, and display them."""
    for fig_idx, (fig, metric_subplots) in enumerate(figures_to_show):
        # Determine the save path for this figure
        current_plot_path = None
        if plot_path is not None:
            if len(figures_to_show) > 1:
                current_plot_path = plot_path / f"plot_fig{fig_idx + 1}.{plot_format}"
            else:
                current_plot_path = plot_path / f"plot.{plot_format}"

        _add_legend_and_save(fig, metric_subplots, current_plot_path)

    # Show all figures at once
    plt.show()


def _organize_metrics_into_groups(
    metrics: list[Metric] | list[list[Metric]],
    individual_plots: bool,
) -> list[list[Metric]]:
    """
    Organize metrics into groups where each group will be plotted in one figure.

    Args:
        metrics: Either a flat list of metrics or a list of lists of metrics
        individual_plots: If True, each metric gets its own figure

    Returns:
        List of metric groups, where each group will be plotted in one figure

    Raises:
        ValueError: If metrics is a list of lists but not all elements are lists

    """
    # Check if metrics is list[list[Metric]]
    if any(isinstance(m, list) for m in metrics):
        if not all(isinstance(m, list) for m in metrics):
            raise ValueError("If metrics is a list of lists, all elements must be lists.")
        if individual_plots:
            # Flatten and make each metric its own group
            flat_metrics: list[Metric] = [m for group in metrics for m in group]  # type: ignore[union-attr]
            return [[metric] for metric in flat_metrics]

        # Use the provided grouping
        return metrics  # type: ignore[return-value]

    # Flat list[Metric]
    flat_metrics_list: list[Metric] = metrics  # type: ignore[assignment]
    if individual_plots:
        # Each metric in its own figure
        return [[metric] for metric in flat_metrics_list]

    # Group into chunks of up to 3 metrics per figure
    groups: list[list[Metric]] = [flat_metrics_list[i : i + 3] for i in range(0, len(flat_metrics_list), 3)]
    return groups


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


def _add_legend_and_save(
    fig: Figure,
    metric_subplots: list[SubPlot],
    plot_path: Path | None = None,
) -> None:
    # Find the first subplot with data to get legend handles
    handles: list[Artist] = []
    labels: list[str] = []
    for sp in metric_subplots:
        handles, labels = sp.get_legend_handles_labels()
        if handles:
            break

    # Only add legend if there are any handles to display
    legend_fig: Figure | None = None
    if handles:
        legend_mode, label_cols, estimated_legend_rows = _select_legend_mode(labels)

        if legend_mode == "same-figure":
            fig.legend(
                handles,
                labels,
                loc="outside upper center",
                ncol=label_cols,
                frameon=True,
            )
        else:
            legend_fig = _create_separate_legend_figure(
                handles,
                labels,
                label_cols=label_cols,
                estimated_rows=estimated_legend_rows,
            )

    if plot_path is not None:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=600)
        LOGGER.info(f"Saved plot to: {plot_path}")

        if legend_fig is not None:
            legend_path = _get_separate_legend_path(plot_path)
            legend_fig.savefig(legend_path, dpi=600)
            LOGGER.info(f"Saved legend to: {legend_path}")


def _select_legend_mode(labels: Sequence[str]) -> tuple[Literal["same-figure", "separate"], int, int]:
    label_count = len(labels)
    max_label_len = max((len(label) for label in labels), default=0)

    max_cols = LEGEND_MAX_COLS
    label_cols = min(max(label_count, 1), max_cols)
    estimated_legend_rows = math.ceil(label_count / label_cols)

    # legend should be plotted as separate figure
    force_separate = (
        label_count >= SEPARATE_LEGEND_MIN_LABELS
        or estimated_legend_rows >= SEPARATE_LEGEND_MIN_ROWS
        or (max_label_len > SEPARATE_LEGEND_MAX_LABEL_LENGTH and label_count >= 6)
    )

    if force_separate:
        return "separate", label_cols, estimated_legend_rows

    # legend should be plotted in the same figure, positioned outside the subplots
    return "same-figure", label_cols, estimated_legend_rows


def _create_separate_legend_figure(
    handles: Sequence[Artist],
    labels: Sequence[str],
    *,
    label_cols: int,
    estimated_rows: int,
) -> Figure:
    # create the legend figure with an initial size guess ...
    initial_width = max(6.0, 1.8 * label_cols)
    initial_height = max(2.0, 0.8 * max(estimated_rows, 1) + 0.8)
    legend_padding_inches = 0.15

    legend_fig = plt.figure(figsize=(initial_width, initial_height))
    legend = legend_fig.legend(handles, labels, loc="center", ncol=label_cols, frameon=True)

    # ... and resize the figure based on the actual legend size
    legend_fig.canvas.draw()
    renderer = legend_fig.canvas.get_renderer()  # type: ignore[attr-defined]
    legend_bbox = legend.get_window_extent(renderer)

    legend_width = legend_bbox.width / legend_fig.dpi
    legend_height = legend_bbox.height / legend_fig.dpi
    fitted_width = legend_width + 2 * legend_padding_inches
    fitted_height = legend_height + 2 * legend_padding_inches

    legend_fig.set_size_inches(fitted_width, fitted_height, forward=True)
    legend_fig.canvas.draw()
    return legend_fig


def _get_separate_legend_path(plot_path: Path) -> Path:
    return plot_path.with_name(f"{plot_path.stem}_legend{plot_path.suffix}")


def _truncate_to_common_finite_prefix(
    data_per_trial: list[Sequence[tuple[X, Y]]],
) -> tuple[list[Sequence[tuple[X, Y]]], bool]:
    truncated_trials: list[Sequence[tuple[X, Y]]] = []
    had_non_finite = False

    for trial_data in data_per_trial:
        finite_prefix: list[tuple[X, Y]] = []
        for point in trial_data:
            x_value, y_value = point
            if not np.isfinite((x_value, y_value)).all().item() or y_value > MAX_LOG_PLOT_VALUE:
                had_non_finite = True
                break
            finite_prefix.append(point)

        if len(finite_prefix) < len(trial_data):
            had_non_finite = True

        if finite_prefix:
            truncated_trials.append(finite_prefix)

    if not truncated_trials:
        return [], had_non_finite

    common_prefix_length = min(len(trial_data) for trial_data in truncated_trials)
    return [trial_data[:common_prefix_length] for trial_data in truncated_trials], had_non_finite


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
        markevery=0.1,
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
    problem: "BenchmarkProblem",
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
