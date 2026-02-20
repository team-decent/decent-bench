import math
import warnings
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes as SubPlot
from matplotlib.figure import Figure

import decent_bench.metrics.metric_utils as utils
from decent_bench.agents import AgentMetricsView
from decent_bench.benchmark_problem import BenchmarkProblem
from decent_bench.distributed_algorithms import P2PAlgorithm
from decent_bench.metrics._computational_cost import ComputationalCost
from decent_bench.metrics._metric import Metric, X, Y
from decent_bench.utils.logger import LOGGER

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


def create_plots(
    resulting_agent_states: dict[P2PAlgorithm, list[list[AgentMetricsView]]],
    problem: BenchmarkProblem,
    metrics: list[Metric] | list[list[Metric]],
    *,
    computational_cost: ComputationalCost | None,
    x_axis_scaling: float = 1e-4,
    compare_iterations_and_computational_cost: bool = False,
    individual_plots: bool = False,
    plot_grid: bool = True,
    plot_path: Path | None = None,
) -> None:
    """
    Plot the execution results with one subplot per metric.

    Each algorithm's curve is its mean across the trials. The surrounding envelope is the min and max across the trials.

    Args:
        resulting_agent_states: resulting agent states from the trial executions, grouped by algorithm
        problem: benchmark problem whose properties, e.g.
            :attr:`~decent_bench.benchmark_problem.BenchmarkProblem.x_optimal`, are used for metric calculations
        metrics: metrics to calculate and plot. If a list of lists is provided, each inner list will be plotted in a
            separate figure. Otherwise groups of 3 metrics will be plotted together in subplots of the same figure
        computational_cost: computational cost settings for plot metrics, if ``None`` x-axis will be iterations instead
            of computational cost
        x_axis_scaling: scaling factor for computational cost x-axis, used to convert the cost units into more
            manageable units for plotting. Only used if ``computational_cost`` is provided
        compare_iterations_and_computational_cost: whether to plot both metric vs computational cost and
            metric vs iterations. Only used if ``computational_cost`` is provided
        individual_plots: whether to create individual plots for each metric instead of subplots
        plot_grid: whether to show grid lines on the plots
        plot_path: optional file path to save the generated plot as an image file (e.g., "plots.png"). If ``None``,
            the plot will only be displayed

    Note:
        Computational cost can be interpreted as the cost of running the algorithm on a specific hardware setup.
        Therefore the computational cost could be seen as the number of operations performed (similar to FLOPS) but
        weighted by the time or energy it takes to perform them on the specific hardware.

        .. include:: snippets/computational_cost.rst

    """
    if not metrics:
        return

    # Normalize metrics into list of groups (each group will be one figure)
    metric_groups = _organize_metrics_into_groups(metrics, individual_plots)

    use_cost = computational_cost is not None
    two_columns = use_cost and compare_iterations_and_computational_cost

    # Track all figures to display at the end
    all_figures: list[tuple[Figure, list[SubPlot]]] = []

    # Create a figure for each metric group
    for metric_group in metric_groups:
        fig, metric_subplots = _create_metric_subplots(
            metric_group,
            use_cost,
            compare_iterations_and_computational_cost,
            plot_grid,
        )
        all_figures.append((fig, metric_subplots))

    # Now plot all the data
    did_plot = False
    with utils.MetricProgressBar() as progress:
        total_plots = (
            sum(len(group) for group in metric_groups) * len(resulting_agent_states) * (2 if two_columns else 1)
        )
        plot_task = progress.add_task(
            "Generating plots",
            total=total_plots,
            status="",
        )
        x_label = X_LABELS["computational_cost" if use_cost else "iterations"]

        for group_idx, metric_group in enumerate(metric_groups):
            fig, metric_subplots = all_figures[group_idx]

            for metric_index_in_group, metric in enumerate(metric_group):
                progress.update(
                    plot_task,
                    status=f"Task: {metric.plot_description} vs {x_label}",
                )
                for alg_idx, (alg, agent_states) in enumerate(resulting_agent_states.items()):
                    data_per_trial: list[Sequence[tuple[X, Y]]] = _plot_data_per_trial(
                        agent_states,
                        problem,
                        metric,
                    )
                    if not _is_finite(data_per_trial):
                        msg = (
                            f"Skipping plot {metric.plot_description}/{x_label} "
                            f"for {alg.name}: found nan or inf in datapoints. "
                            f"Test data or optimal x may be missing from the benchmark problem, got: "
                            f"test_data={type(problem.test_data)}, x_optimal={type(problem.x_optimal)}"
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
                        metric_index_in_group,
                        alg_idx,
                    )
                    did_plot = True
                    progress.advance(plot_task, 2 if two_columns else 1)
        progress.update(plot_task, status="Finalizing plots")

    if not did_plot:
        LOGGER.warning("No plots were generated due to invalid data.")
        return

    # Filter out empty figures (ones with no data plotted in any subplot)
    figures_to_show = [
        (fig, metric_subplots)
        for fig, metric_subplots in all_figures
        if any(sp.get_legend_handles_labels()[0] for sp in metric_subplots)  # Check if any subplot has data
    ]

    if not figures_to_show:
        LOGGER.warning("All figures are empty, nothing to display.")
        return

    # Add legends and save all non-empty figures
    for fig_idx, (fig, metric_subplots) in enumerate(figures_to_show):
        # Append figure number to plot_path if there are multiple figures
        current_plot_path = plot_path
        if plot_path is not None and len(figures_to_show) > 1:
            # Split the path into name and extension
            current_plot_path = plot_path.with_stem(f"{plot_path.stem}_fig{fig_idx + 1}")

        _add_legend_and_save(fig, metric_subplots, two_columns, current_plot_path)

    # Close empty figures to free memory
    for fig, metric_subplots in all_figures:
        if (fig, metric_subplots) not in figures_to_show:
            plt.close(fig)

    # Show all non-empty figures at once
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
    two_columns: bool,
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
    if handles:
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
        fig.savefig(plot_path, dpi=600)
        LOGGER.info(f"Saved plot to: {plot_path}")


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
    alg: P2PAlgorithm,
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
