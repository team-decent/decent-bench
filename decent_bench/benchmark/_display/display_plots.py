import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.artist import Artist
from matplotlib.axes import Axes as SubPlot
from matplotlib.figure import Figure

from decent_bench.metrics._computational_cost import ComputationalCost
from decent_bench.metrics._metric import Metric
from decent_bench.metrics._metrics_view import NetworkMetricsView
from decent_bench.utils.logger import LOGGER

if TYPE_CHECKING:
    from decent_bench.benchmark import MetricResult

MIN_Y_VALUE = 1e-15  # replacement for negative/zero values when y_log
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
    """Display and optionally save plot metrics."""
    # process y and x values (remove inf, replace non-positive values, add compute cost)
    frames_by_metrics = _preprocess(
        metrics_result, computational_cost, scale_x_axis, compare_iterations_and_computational_cost
    )
    # the result is a dict[Metric, DataFrame]
    # each DataFrame has column (algorithm, mean, min, max) and for x axis:
    #    "iteration" only (if no computational cost)
    #    "compute" only (if only computational cost is to be plotted)
    #    otherwise "iteration" and "compute" both

    all_figures = _create_and_plot_figures(frames_by_metrics, plot_grid, individual_plots)

    if not all_figures:
        LOGGER.warning("No plots were generated due to invalid data.")
        return

    _save_and_show_figures(all_figures, plot_path=plot_path, plot_format=plot_format, show_plots=show_plots)


def _preprocess(
    metrics_result: "MetricResult",
    computational_cost: ComputationalCost | None,
    scale_x_axis: float = 1e-4,
    compare_iterations_and_computational_cost: bool = False,
) -> dict[Metric, pd.DataFrame]:
    if metrics_result.raw_plot_results is None or metrics_result.plot_results is None:
        return {}

    plot_results = metrics_result.plot_results
    metrics = list(metrics_result.raw_plot_results.keys())

    # 1) split into dict[Metric, DataFrame] (copies data to avoid modifying original)
    frames = {
        metric: plot_results[plot_results["metric"] == metric.description].copy().drop(columns=["metric"])
        for metric in metrics
    }

    for metric, frame in frames.items():
        # 2) truncate at first occurrence of inf in mean/min/max (likely divergence)

        has_inf = frame[["mean", "min", "max"]].isin([np.inf, -np.inf]).any(axis=1)
        if has_inf.any():
            rows_with_inf = frame[has_inf]
            cutoff = rows_with_inf.groupby("algorithm", observed=False)["iteration"].min().rename("cutoff")
            # map cutoff iteration to the corresponding algorithm
            frame["cutoff"] = frame.set_index("algorithm").index.map(cutoff.to_dict())

            # truncate: keep a row if no inf was found or, if found, if iteration < cutoff
            frames[metric] = (
                frame[(frame["cutoff"].isna()) | (frame["iteration"] < frame["cutoff"])].copy().drop(columns=["cutoff"])
            )

        # 3) replace negative values if y_log
        if metric.y_log:
            subset = frames[metric][["mean", "min", "max"]]
            if (subset <= 0).any().any():
                positive_values = subset[subset > 0].min().min()
                non_positive_replacement = positive_values.min() if not positive_values else MIN_Y_VALUE
                frames[metric][["mean", "min", "max"]] = subset.mask(subset <= 0, non_positive_replacement)

                flat = subset.melt(value_name="val")["val"]
                unique_non_positive = flat[flat <= 0].unique()
                LOGGER.warning(
                    f"Metric '{metric.description}' has y_log=True but contains non-positive y values. "
                    f"They were replaced with {non_positive_replacement} for plotting purposes. "
                    f"Non-positive values that were replaced (when close to 0, they are likely rounding errors): "
                    f"{unique_non_positive.tolist()}."
                )

        # 4) increment iteration by 1 if x_log
        if metric.x_log and np.any(frames[metric]["iteration"] == 0):
            frames[metric]["iteration"] += 1
            LOGGER.warning(
                f"Metric '{metric.description}' has x_log=True but contains iteration=0. "
                f"Shifted all x values for plotting purposes."
            )

    # 5) resolve handling of x-axis
    # if computational_cost is None -> skip adding column "compute"
    #    if also compare_iterations_and_computational_cost=True -> warning, otherwise silent
    # if computational_cost provided, but no network_views -> skip "compute" and warn
    # if computational_cost and network_views -> add "compute" column
    #    and drop "iteration" if compare_iterations_and_computational_cost=False
    if computational_cost is None:
        if compare_iterations_and_computational_cost:
            LOGGER.warning(
                "``computational_cost`` is None, cannot compute "
                "total computational cost. Plotting against iterations instead."
            )
        return frames
    if metrics_result.network_views is None:
        LOGGER.warning(
            "``computational_cost`` provided but network views are missing. Cannot compute "
            "total computational cost. Plotting against iterations instead."
        )
        return frames

    # get scaling factors
    scaling = {
        algorithm.name: _calc_total_cost(list(network_views), computational_cost) * scale_x_axis
        for algorithm, network_views in metrics_result.network_views.items()
    }

    # add "compute" column (scaled according to computational cost and scale_x_axis)
    for metric, frame in frames.items():
        frame["compute"] = frame["iteration"] * frame["algorithm"].map(scaling).astype(float)
        if not compare_iterations_and_computational_cost:
            frames[metric] = frame.drop(columns=["iteration"])

    return frames


def _calc_total_cost(network_views: list[NetworkMetricsView], computational_cost: ComputationalCost) -> float:
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


def _create_and_plot_figures(
    frames_by_metrics: dict[Metric, pd.DataFrame],
    plot_grid: bool,
    individual_plots: bool,
) -> list[tuple[Figure, list[SubPlot]]]:
    """Create figures, plot data, and return non-empty figures."""
    if not frames_by_metrics:
        return []

    metric_groups = _organize_metrics_into_groups(list(frames_by_metrics.keys()), individual_plots)

    sample_df = next(iter(frames_by_metrics.values()))

    algs = list(sample_df["algorithm"].unique())
    use_cost, use_iteration = "compute" in sample_df, "iteration" in sample_df
    two_columns = use_cost and use_iteration

    all_figures: list[tuple[Figure, list[SubPlot]]] = []

    for metric_group in metric_groups:
        fig, metric_subplots = _create_metric_subplots(
            metric_group,
            use_cost,
            two_columns,
            plot_grid,
        )
        all_figures.append((fig, metric_subplots))

    for group_idx, metric_group in enumerate(metric_groups):
        fig, metric_subplots = all_figures[group_idx]

        for metric_index_in_group, metric in enumerate(metric_group):
            for alg_idx, alg_name in enumerate(algs):
                frame = frames_by_metrics[metric].loc[frames_by_metrics[metric]["algorithm"] == alg_name]
                y_mean, y_min, y_max = frame["mean"].tolist(), frame["min"].tolist(), frame["max"].tolist()

                subplot_refs = (
                    (metric_subplots[metric_index_in_group * 2], metric_subplots[metric_index_in_group * 2 + 1])
                    if two_columns
                    else (metric_subplots[metric_index_in_group],)
                )
                x_axes = (
                    (frame["compute"], frame["iteration"])
                    if two_columns
                    else (frame["compute"] if use_cost else frame["iteration"],)
                )

                for subplot, x in zip(subplot_refs, x_axes, strict=True):
                    _plot_subplot(subplot, x.tolist(), y_mean, y_min, y_max, alg_name, alg_idx)

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
