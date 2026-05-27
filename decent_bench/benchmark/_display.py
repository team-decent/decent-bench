import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd
from rich.status import Status

from decent_bench.benchmark._metric_result import MetricResult
from decent_bench.metrics import ComputationalCost, Metric
from decent_bench.metrics.plots.display_plots import display_plots
from decent_bench.metrics.tables.display_tables import display_tables
from decent_bench.utils import logger
from decent_bench.utils.logger import LOGGER

if TYPE_CHECKING:
    from decent_bench.algorithms import Algorithm
    from decent_bench.networks import Network
    from decent_bench.utils.checkpoint_manager import CheckpointManager


def display_metrics(  # noqa: PLR0912
    metrics_result: MetricResult | None = None,
    checkpoint_manager: "CheckpointManager | None" = None,
    *,
    table_metrics: list[Metric | str] | None = None,
    plot_metrics: list[Metric | str] | None = None,
    algorithms: list["Algorithm[Network] | str"] | None = None,
    table_fmt: Literal["grid", "latex"] = "grid",
    plot_grid: bool = True,
    individual_plots: bool = False,
    computational_cost: ComputationalCost | None = None,
    scale_x_axis: float = 1e-4,
    scale_compute: float = 1.0,
    compare_iterations_and_computational_cost: bool = False,
    save_path: str | Path | None = None,
    plot_format: Literal["png", "pdf", "svg"] = "png",
    show_plots: bool = True,
    log_level: int = logging.INFO,
) -> None:
    """
    Display metrics from a metrics result.

    Args:
        metrics_result: result of metrics computation containing the metrics to display. If not provided,
            the result will be loaded from the checkpoint manager
        checkpoint_manager: if provided, will be used to load metrics result.
        table_metrics: metrics to tabulate, defaults to ``None`` which will display all metrics in the metrics_result.
            Entries can be :class:`~decent_bench.metrics.Metric` objects or strings
            (matching :attr:`~decent_bench.metrics.Metric.description`).
        plot_metrics: metrics to plot, defaults to ``None`` which will display all metrics in the metrics_result.
            Entries can be :class:`~decent_bench.metrics.Metric` objects or strings
            (matching :attr:`~decent_bench.metrics.Metric.description`).
            If ``individual_plots`` is True, each metric is plotted in its own figure;
            otherwise a maximum of 3 metrics are plotted as subplots in the same figure.
        algorithms: algorithms to display. If provided, only these algorithms are included in tables and plots.
            Entries can be :class:`~decent_bench.algorithms.Algorithm` objects or strings
            (matching :attr:`~decent_bench.algorithms.Algorithm.name`).
            Defaults to ``None`` which displays all algorithms in the metrics_result.
        table_fmt: table format, grid is suitable for the terminal while latex can be copy-pasted into a latex document
        plot_grid: whether to show grid lines on the plots
        individual_plots: whether to plot each metric in a separate figure
        computational_cost: computational cost settings for plot metrics, if ``None`` x-axis will be iterations instead
            of computational cost
        scale_x_axis: scaling factor for computational cost x-axis, used to convert the cost units into more
            manageable units for plotting. Only used if ``computational_cost`` is provided.
        scale_compute: scaling factor for the compute related metrics (i.e.
            :class:`~decent_bench.metrics.metric_library.FunctionCalls`,
            :class:`~decent_bench.metrics.metric_library.GradientCalls`,
            :class:`~decent_bench.metrics.metric_library.HessianCalls` and
            :class:`~decent_bench.metrics.metric_library.ProximalCalls`) shown in the table, used to convert the
            raw count into more manageable units for display.
        compare_iterations_and_computational_cost: whether to plot both metric vs computational cost and
            metric vs iterations. Only used if ``computational_cost`` is provided.
        save_path: optional directory path to save the tables and plots to. Tables will be saved as ``table.txt`` and
            ``table.tex`` while plots will be saved as ``plot_{#}.{format}`` in the specified directory.
            If checkpoint_manager is provided then the default save path will be the results path in the checkpoint
            manager, which is determined by
            :meth:`~decent_bench.utils.checkpoint_manager.CheckpointManager.get_results_path`. If both are provided,
            the provided ``save_path`` will be used. If neither a checkpoint manager or a save path is provided,
            the tables and plots are not saved to disk.
        plot_format: format to save plots in, defaults to ``png``. Can be ``png``, ``pdf``, or ``svg``.
        show_plots: whether to show the plots after creating them, defaults to ``True``. Can be useful to set to
            ``False`` when running in a non-interactive environment or when only saving the plots without displaying.
        log_level: minimum level to log, e.g. :data:`logging.INFO`

    Raises:
        ValueError: If neither ``metrics_result`` nor ``checkpoint_manager`` is provided, or
            if the checkpoint manager does not contain a valid metrics result to load. Also raised if
            ``algorithms`` filtering results in no algorithms remaining, or if both ``table_metrics`` and
            ``plot_metrics`` are specified and both result in no metrics remaining after filtering.
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
    LOGGER.info("Displaying metrics")

    if metrics_result is None:
        if checkpoint_manager is None:
            raise ValueError(
                "If ``metrics_result`` is not provided, ``checkpoint_manager`` must be provided "
                "to load the metrics result from."
            )
        try:
            with Status("Loading metrics result from checkpoint manager..."):
                metrics_result = checkpoint_manager.load_metrics_result()
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Metrics result file not found in checkpoint manager: {e}") from e

        if metrics_result is None:
            raise ValueError("No metrics result found in checkpoint manager to display")

    # filter `metrics_result` based on `plot_metrics`, `table_metrics`, and `algorithms` (if provided)
    prev_values = {
        "network_views": metrics_result.network_views,
        "raw_table_results": metrics_result.raw_table_results,
        "raw_plot_results": metrics_result.raw_plot_results,
        "table_results": metrics_result.table_results,
        "plot_results": metrics_result.plot_results,
        "table_metrics": metrics_result.table_metrics,
        "plot_metrics": metrics_result.plot_metrics,
    }

    if table_metrics is not None:
        metrics_result.table_metrics = _get_new_metrics(metrics_result.table_metrics, table_metrics, "table")

    if plot_metrics is not None:
        metrics_result.plot_metrics = _get_new_metrics(metrics_result.plot_metrics, plot_metrics, "plot")

    if algorithms is not None:
        metrics_result = _filter_algorithms(metrics_result, algorithms)

    # check that filtering didn't empty out the displayable results
    if algorithms is not None and not metrics_result.available_algorithms:
        raise ValueError(
            f"No algorithms remain after filtering. Requested algorithms not found in metrics result. "
            f"Available algorithms: {', '.join(metrics_result.available_algorithms)}"
        )

    if (table_metrics is not None and not metrics_result.available_table_metrics) and (
        plot_metrics is not None and not metrics_result.available_plot_metrics
    ):
        raise ValueError(
            f"No table or plot metrics remain after filtering. "
            f"Available table metrics: {', '.join(metrics_result.available_table_metrics)}. "
            f"Available plot metrics: {', '.join(metrics_result.available_plot_metrics)}"
        )

    if save_path is not None:
        save_path = Path(save_path)
    elif save_path is None and checkpoint_manager is not None:
        save_path = checkpoint_manager.get_results_path()

    if metrics_result.table_metrics:
        display_tables(metrics_result, table_fmt=table_fmt, scale_compute=scale_compute, table_path=save_path)
    else:
        LOGGER.warning("No table metrics to display, skipping tables")

    if metrics_result.plot_metrics:
        display_plots(
            metrics_result,
            computational_cost=computational_cost,
            scale_x_axis=scale_x_axis,
            compare_iterations_and_computational_cost=compare_iterations_and_computational_cost,
            individual_plots=individual_plots,
            plot_grid=plot_grid,
            plot_format=plot_format,
            plot_path=save_path,
            show_plots=show_plots,
        )
    else:
        LOGGER.warning("No plot metrics to display, skipping plots")

    for attribute, prev_value in prev_values.items():
        setattr(metrics_result, attribute, prev_value)


def _filter_algorithms(
    metrics_result: MetricResult,
    algorithms: list["Algorithm[Network] | str"],
) -> MetricResult:
    selected_names = {_get_name_or_value(algorithm, "name") for algorithm in algorithms}
    available_names = set(metrics_result.available_algorithms)
    missing_names = sorted(selected_names - available_names)
    if missing_names:
        LOGGER.warning(
            "Requested algorithm(s) not found in metrics result, skipping: "
            f"{', '.join(missing_names)}. Available algorithms: {', '.join(metrics_result.available_algorithms)}"
        )

    if metrics_result.network_views is not None:
        metrics_result.network_views = {
            algorithm: values
            for algorithm, values in metrics_result.network_views.items()
            if algorithm.name in selected_names
        }

    for attribute in ("table_results", "plot_results"):
        frame = getattr(metrics_result, attribute, None)
        if frame:
            mask = frame.index.get_level_values("algorithm").isin(selected_names)
            setattr(metrics_result, attribute, frame[mask])

    if metrics_result.raw_table_results is not None:
        filtered_raw_table_results: dict[Metric, object] = {}
        for metric, frame in metrics_result.raw_table_results.items():
            if not hasattr(frame.index, "names") or "algorithm" not in frame.index.names:
                filtered_raw_table_results[metric] = frame
                continue
            selected_frame = frame[frame.index.get_level_values("algorithm").isin(selected_names)]
            filtered_raw_table_results[metric] = selected_frame
        metrics_result.raw_table_results = filtered_raw_table_results

    if metrics_result.raw_plot_results is not None:
        filtered_raw_plot_results: dict[Metric, object] = {}
        for metric, frame in metrics_result.raw_plot_results.items():
            if not hasattr(frame.index, "names") or "algorithm" not in frame.index.names:
                filtered_raw_plot_results[metric] = frame
                continue
            selected_frame = frame[frame.index.get_level_values("algorithm").isin(selected_names)]
            filtered_raw_plot_results[metric] = selected_frame
        metrics_result.raw_plot_results = filtered_raw_plot_results

    return metrics_result


def _get_new_metrics(
    metrics: list[Metric] | None,
    requested_metrics: list[Metric | str],
    type_: Literal["table", "plot"],
) -> list[Metric]:
    if metrics is None:
        return []

    lookup = _build_metric_lookup(metrics)

    new_metrics: list[Metric] = []
    for metric_or_name in requested_metrics:
        metric_name = _get_name_or_value(metric_or_name, "description")
        original_metric = lookup.get(metric_name)
        if original_metric is None:
            LOGGER.warning(f"Requested {type_} metric '{metric_name}' not found in metrics result, skipping")
            continue
        new_metrics.append(original_metric)

    return new_metrics


def _build_metric_lookup(metrics: list[Metric]) -> dict[str, Metric]:
    lookup: dict[str, Metric] = {}
    for metric in metrics:
        lookup.setdefault(metric.description, metric)

    return lookup


def _get_name_or_value(value: object, attribute: str) -> str:
    return value if isinstance(value, str) else str(getattr(value, attribute))
