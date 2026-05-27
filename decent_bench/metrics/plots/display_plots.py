import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.artist import Artist
from matplotlib.axes import Axes as SubPlot
from matplotlib.figure import Figure

from decent_bench.algorithms import Algorithm
from decent_bench.metrics._computational_cost import ComputationalCost
from decent_bench.metrics._metric import Metric
from decent_bench.metrics._metrics_view import NetworkMetricsView
from decent_bench.networks import Network
from decent_bench.utils.logger import LOGGER

if TYPE_CHECKING:
    from decent_bench.benchmark import MetricResult


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
# thresholds to decide when to plot the legend as a separate figure
LEGEND_MAX_COLS = 3
SEPARATE_LEGEND_MIN_LABELS = 9
SEPARATE_LEGEND_MIN_ROWS = 4
SEPARATE_LEGEND_MAX_LABEL_LENGTH = 24
# max number of subplots in a figure
MAX_NUM_SUBPLOTS = 3


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
    show_plots: bool = True,
) -> None:
    """Display and optionally save metric plots."""
    plot_results = metrics_result.plot_results
    metrics = metrics_result.plot_metrics
    network_views = metrics_result.network_views

    if plot_results is None or metrics is None:
        return

    if isinstance(plot_results, pd.DataFrame):
        if plot_results.empty:
            return
        if not isinstance(plot_results.index, pd.MultiIndex):
            LOGGER.warning("plot_results must use a MultiIndex to be displayed")
            return
        plot_lookup = _build_plot_lookup(plot_results)

    metric_groups = _organize_metrics_into_groups(metrics, individual_plots)

    use_cost = computational_cost is not None
    two_columns = use_cost and compare_iterations_and_computational_cost

    all_figures = _create_and_plot_figures(
        metric_groups,
        plot_lookup,
        network_views,
        use_cost,
        two_columns,
        computational_cost=computational_cost,
        scale_x_axis=scale_x_axis,
        plot_grid=plot_grid,
    )

    if not all_figures:
        LOGGER.warning("No plots were generated due to invalid data.")
        return

    _save_and_show_figures(all_figures, plot_path=plot_path, plot_format=plot_format, show_plots=show_plots)


def _create_and_plot_figures(  # noqa: PLR0912
    metric_groups: list[list[Metric]],
    plot_results: Mapping[str, Mapping[str, tuple[Sequence[float], Sequence[float], Sequence[float], Sequence[float]]]],
    resulting_network_views: Mapping[Algorithm[Network], Sequence[NetworkMetricsView]] | None,
    use_cost: bool,
    two_columns: bool,
    *,
    computational_cost: ComputationalCost | None,
    scale_x_axis: float,
    plot_grid: bool,
) -> list[tuple[Figure, list[SubPlot]]]:
    """Create figures, plot data, and return non-empty figures."""
    all_figures: list[tuple[Figure, list[SubPlot]]] = []

    for metric_group in metric_groups:
        fig, metric_subplots = _create_metric_subplots(
            metric_group,
            use_cost,
            two_columns,
            plot_grid,
        )
        all_figures.append((fig, metric_subplots))

    algs = list(plot_results.keys())
    network_views_by_name = {
        algorithm.name: list(network_views)
        for algorithm, network_views in (resulting_network_views or {}).items()
    }

    for group_idx, metric_group in enumerate(metric_groups):
        fig, metric_subplots = all_figures[group_idx]

        for metric_index_in_group, metric in enumerate(metric_group):
            metric_name = metric.description
            for alg_idx, alg_name in enumerate(algs):
                if metric_name not in plot_results[alg_name]:
                    continue

                x, y_mean, y_min, y_max = plot_results[alg_name][metric_name]

                if metric.x_log and any(val <= 0 for val in x):
                    offset = 1 - min(x)
                    x = [val + offset for val in x]
                    LOGGER.warning(
                        f"Metric '{metric.description}' has x_log=True but contains non-positive x values. "
                        f"Added {offset} to all x values for plotting purposes."
                    )
                if metric.y_log and any(val <= 0 for val in y_mean):
                    non_positive_replacement = min(*(val for val in y_mean if val > 0), 1e-8)
                    y_mean = [val if val > 0 else non_positive_replacement for val in y_mean]
                    LOGGER.warning(
                        f"Metric '{metric.description}' has y_log=True but contains non-positive y values for"
                        f"algorithm {alg_name}. These values have been replaced with {non_positive_replacement} for "
                        f"plotting purposes."
                    )
                if metric.y_log and any(val <= 0 for val in y_min):
                    non_positive_replacement = min(*(val for val in y_min if val > 0), 1e-8)
                    y_min = [val if val > 0 else non_positive_replacement for val in y_min]
                if metric.y_log and any(val <= 0 for val in y_max):
                    non_positive_replacement = min(*(val for val in y_max if val > 0), 1e-8)
                    y_max = [val if val > 0 else non_positive_replacement for val in y_max]

                subplot_idx = metric_index_in_group * (2 if two_columns else 1)

                x_to_plot = x
                if use_cost and computational_cost is not None:
                    if resulting_network_views is None:
                        LOGGER.warning(
                            f"Computational cost provided but resulting network views are missing. Cannot compute "
                            f"total computational cost for algorithm {alg_name}. Plotting against iterations instead."
                        )
                    else:
                        network_views_for_alg = network_views_by_name.get(alg_name)
                        if network_views_for_alg is None:
                            LOGGER.warning(
                                f"No network views available for algorithm {alg_name}. "
                                "Plotting against iterations instead of computational cost."
                            )
                            network_views_for_alg = []
                        total_computational_cost = _calc_total_cost(network_views_for_alg, computational_cost)
                        x_to_plot = tuple(val * total_computational_cost * scale_x_axis for val in x)

                if two_columns:
                    iter_idx = metric_index_in_group * 2 + 1
                    _plot_subplot(metric_subplots[iter_idx], x, y_mean, y_min, y_max, alg_name, alg_idx)

                _plot_subplot(metric_subplots[subplot_idx], x_to_plot, y_mean, y_min, y_max, alg_name, alg_idx)

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
    show_plots: bool,
) -> None:
    """Add legends, save figures to files, and display them."""
    for fig_idx, (fig, metric_subplots) in enumerate(figures_to_show):
        current_plot_path = None
        if plot_path is not None:
            if len(figures_to_show) > 1:
                current_plot_path = plot_path / f"plot_fig{fig_idx + 1}.{plot_format}"
            else:
                current_plot_path = plot_path / f"plot.{plot_format}"

        _add_legend_and_save(fig, metric_subplots, current_plot_path)

    if show_plots:
        plt.show()
    else:
        for fig, _ in figures_to_show:
            plt.close(fig)


def _organize_metrics_into_groups(
    metrics: list[Metric],
    individual_plots: bool,
) -> list[list[Metric]]:
    if individual_plots:
        return [[metric] for metric in metrics]

    return [metrics[i : i + MAX_NUM_SUBPLOTS] for i in range(0, len(metrics), MAX_NUM_SUBPLOTS)]


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

        if i // n_cols == n_rows - 1:
            if n_cols == 2:
                sp.set_xlabel(X_LABELS["iterations"] if i % 2 == 1 else X_LABELS["computational_cost"])
            else:
                sp.set_xlabel(X_LABELS["computational_cost" if use_cost else "iterations"])

        if i % n_cols == 0:
            sp.set_ylabel(metric.description)

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
    handles: list[Artist] = []
    labels: list[str] = []
    for sp in metric_subplots:
        handles, labels = sp.get_legend_handles_labels()
        if handles:
            break

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

    force_separate = (
        label_count >= SEPARATE_LEGEND_MIN_LABELS
        or estimated_legend_rows >= SEPARATE_LEGEND_MIN_ROWS
        or (max_label_len > SEPARATE_LEGEND_MAX_LABEL_LENGTH and label_count >= 6)
    )

    if force_separate:
        return "separate", label_cols, estimated_legend_rows

    return "same-figure", label_cols, estimated_legend_rows


def _create_separate_legend_figure(
    handles: Sequence[Artist],
    labels: Sequence[str],
    *,
    label_cols: int,
    estimated_rows: int,
) -> Figure:
    initial_width = max(6.0, 1.8 * label_cols)
    initial_height = max(2.0, 0.8 * max(estimated_rows, 1) + 0.8)
    legend_padding_inches = 0.15

    legend_fig = plt.figure(figsize=(initial_width, initial_height))
    legend = legend_fig.legend(handles, labels, loc="center", ncol=label_cols, frameon=True)

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


def _plot_subplot(
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
    n_combinations = len(MARKERS) * len(STYLES)

    idx = index % n_combinations

    color_idx = index % len(COLORS)
    marker_idx = idx % len(MARKERS)
    style_idx = (idx // len(MARKERS)) % len(STYLES)

    return MARKERS[marker_idx], STYLES[style_idx], COLORS[color_idx]


def _calc_total_cost(network_views: list[NetworkMetricsView], computational_cost: ComputationalCost) -> float:
    if not network_views:
        return 0.0

    non_server_states = [a for network_view in network_views for a in network_view.agents()]
    mean_function_calls = np.mean([a.n_function_calls for a in non_server_states])
    mean_gradient_calls = np.mean([a.n_gradient_calls for a in non_server_states])
    mean_hessian_calls = np.mean([a.n_hessian_calls for a in non_server_states])
    mean_proximal_calls = np.mean([a.n_proximal_calls for a in non_server_states])
    mean_communication_calls = np.mean([a.n_sent_messages for a in non_server_states])

    return float(
        computational_cost.function * mean_function_calls
        + computational_cost.gradient * mean_gradient_calls
        + computational_cost.hessian * mean_hessian_calls
        + computational_cost.proximal * mean_proximal_calls
        + computational_cost.communication * mean_communication_calls
    )


def _build_plot_lookup(
    plot_results: pd.DataFrame,
) -> dict[str, dict[str, tuple[Sequence[float], Sequence[float], Sequence[float], Sequence[float]]]]:
    lookup: dict[str, dict[str, tuple[Sequence[float], Sequence[float], Sequence[float], Sequence[float]]]] = {}

    if not isinstance(plot_results.index, pd.MultiIndex):
        return lookup

    grouped = plot_results.groupby(level=["metric", "algorithm"], sort=False)
    for (metric_name, algorithm_name), group in grouped:
        ordered = group.sort_index(level="iterations")
        index_vals = ordered.index.get_level_values("iterations")
        x_values = index_vals.to_numpy(dtype=float).tolist()
        y_mean = ordered["y_mean"].to_numpy(dtype=float).tolist()
        y_min = ordered["y_min"].to_numpy(dtype=float).tolist()
        y_max = ordered["y_max"].to_numpy(dtype=float).tolist()
        lookup.setdefault(str(algorithm_name), {})[str(metric_name)] = (x_values, y_mean, y_min, y_max)

    return lookup


