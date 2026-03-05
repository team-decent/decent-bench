import logging
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from decent_bench.benchmark._metric_result import MetricResult
from decent_bench.metrics import (
    ComputationalCost,
    Metric,
    display_plots,
    display_tables,
)
from decent_bench.utils import logger
from decent_bench.utils.logger import LOGGER

if TYPE_CHECKING:
    from decent_bench.utils.checkpoint_manager import CheckpointManager


def display_metrics(
    metrics_result: MetricResult | None = None,
    checkpoint_manager: "CheckpointManager | None" = None,
    *,
    table_metrics: list[Metric] | None = None,
    plot_metrics: list[Metric] | list[list[Metric]] | None = None,
    table_fmt: Literal["grid", "latex"] = "grid",
    plot_grid: bool = True,
    individual_plots: bool = False,
    computational_cost: ComputationalCost | None = None,
    x_axis_scaling: float = 1e-4,
    compare_iterations_and_computational_cost: bool = False,
    save_path: str | Path | None = None,
    plot_format: Literal["png", "pdf", "svg"] = "png",
    log_level: int = logging.INFO,
) -> None:
    """
    Display metrics from a metrics result.

    Args:
        metrics_result: result of metrics computation containing the metrics to display. If not provided,
            the result will be loaded from the checkpoint manager
        checkpoint_manager: if provided, will be used to load metrics result.
        table_metrics: metrics to tabulate, defaults to ``None`` which will display all metrics in the metrics_result.
        plot_metrics: metrics to plot, defaults to ``None`` which will display all metrics in the metrics_result.
            If a list of lists is provided, each inner list will be plotted in a separate figure. Otherwise up to 3
            metrics will be grouped and plotted in their own figure with subplots.
        table_fmt: table format, grid is suitable for the terminal while latex can be copy-pasted into a latex document
        plot_grid: whether to show grid lines on the plots
        individual_plots: whether to plot each metric in a separate figure
        computational_cost: computational cost settings for plot metrics, if ``None`` x-axis will be iterations instead
            of computational cost
        x_axis_scaling: scaling factor for computational cost x-axis, used to convert the cost units into more
            manageable units for plotting. Only used if ``computational_cost`` is provided.
        compare_iterations_and_computational_cost: whether to plot both metric vs computational cost and
            metric vs iterations. Only used if ``computational_cost`` is provided.
        save_path: optional directory path to save the tables and plots to. Tables will be saved as ``table.txt`` and
            ``table.tex`` while plots will be saved as ``plot_{#}.{format}`` in the specified directory.
            If not provided, the tables and plots are not saved to files.
        plot_format: format to save plots in, defaults to ``png``. Can be ``png``, ``pdf``, or ``svg``.
        log_level: minimum level to log, e.g. :data:`logging.INFO`

    Raises:
        ValueError: If neither ``metrics_result`` nor ``checkpoint_manager`` is provided, or
            if the checkpoint manager does not contain a valid metrics result to load.
        FileNotFoundError: If ``metrics_result`` is not provided and the checkpoint manager does not contain a metrics
            result file to load.

    Note:
        Checkpoint_manager is ignored if ``metrics_result`` is provided. If either ``plot_metrics`` or ``table_metrics``
        is provided, only metrics which are present in the provided ``metrics_result`` will be displayed.
        If neither is provided, all metrics in the ``metrics_result`` will be displayed.

        Computational cost can be interpreted as the cost of running the algorithm on a specific hardware setup.
        Therefore the computational cost could be seen as the number of operations performed (similar to FLOPS) but
        weighted by the time or energy it takes to perform them on the specific hardware.

        .. include:: snippets/computational_cost.rst

        If ``computational_cost`` is provided and ``compare_iterations_and_computational_cost`` is ``True``, each metric
        will be plotted twice: once against computational cost and once against iterations.
        Computational cost plots will be shown on the left and iteration plots on the right.

    """
    logger.start_logger(log_level=log_level)

    if metrics_result is None:
        if checkpoint_manager is None:
            raise ValueError(
                "If ``metrics_result`` is not provided, ``checkpoint_manager`` must be provided "
                "to load the metrics result from."
            )
        try:
            metrics_result = checkpoint_manager.load_metrics_result()
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Metrics result file not found in checkpoint manager: {e}") from e

        if metrics_result is None:
            raise ValueError("No metrics result found in checkpoint manager to display")

    # Update metrics to display based on provided plot_metrics and table_metrics (if provided)
    new_metrics_result = deepcopy(metrics_result)

    if table_metrics is not None:
        new_metrics_result.table_metrics = _get_new_table_metrics(new_metrics_result, table_metrics)

    if plot_metrics is not None:
        new_metrics_result.plot_metrics = _get_new_plot_metrics(new_metrics_result, plot_metrics)

    save_path = Path(save_path) if save_path is not None else None
    display_tables(new_metrics_result, table_fmt=table_fmt, table_path=save_path)
    display_plots(
        new_metrics_result,
        computational_cost=computational_cost,
        x_axis_scaling=x_axis_scaling,
        compare_iterations_and_computational_cost=compare_iterations_and_computational_cost,
        individual_plots=individual_plots,
        plot_grid=plot_grid,
        plot_format=plot_format,
        plot_path=save_path,
    )


def _get_new_table_metrics(
    metrics_result: MetricResult,
    table_metrics: list[Metric],
) -> list[Metric]:
    if metrics_result.table_metrics is None:
        return []

    new_table_metrics = []
    for metric in table_metrics:
        # Find the original metric object in the metrics result that matches the requested metric description,
        # and use that one to prevent KeyErrors in the computed metrics dict
        original_metric = next(
            (m for m in metrics_result.table_metrics if m.table_description == metric.table_description),
            None,
        )
        if original_metric is None:
            LOGGER.warning(f"Requested table metric '{metric.table_description}' not found in metrics result, skipping")
            continue
        new_table_metrics.append(original_metric)

    return new_table_metrics


def _get_new_plot_metrics(
    metrics_result: MetricResult,
    plot_metrics: list[Metric] | list[list[Metric]],
) -> list[Metric] | list[list[Metric]]:
    if metrics_result.plot_metrics is None:
        return []

    flat_metrics: list[Metric] = []
    if any(isinstance(m, list) for m in metrics_result.plot_metrics):
        flat_metrics = [metric for group in metrics_result.plot_metrics for metric in group]  # type: ignore[union-attr]
    else:
        flat_metrics = metrics_result.plot_metrics  # type: ignore[assignment]

    new_plot_metrics = []
    for group in plot_metrics:
        if isinstance(group, list):
            new_group = []
            for metric in group:
                original_metric = next(
                    (m for m in flat_metrics if m.plot_description == metric.plot_description),
                    None,
                )
                if original_metric is None:
                    LOGGER.warning(
                        f"Requested plot metric '{metric.plot_description}' not found in metrics result, skipping"
                    )
                    continue
                new_group.append(original_metric)
            if new_group:
                new_plot_metrics.append(new_group)
        else:
            original_metric = next(
                (m for m in flat_metrics if m.plot_description == group.plot_description),
                None,
            )
            if original_metric is None:
                LOGGER.warning(
                    f"Requested plot metric '{group.plot_description}' not found in metrics result, skipping"
                )
                continue
            new_plot_metrics.append(original_metric)  # type: ignore[arg-type]

    return new_plot_metrics
