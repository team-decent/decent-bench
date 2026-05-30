import copy
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd
from rich.status import Status

from decent_bench.benchmark._display.display_plots import display_plots
from decent_bench.benchmark._display.display_tables import display_tables
from decent_bench.benchmark._metric_result import MetricResult
from decent_bench.metrics import ComputationalCost, Metric
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
    table_fmt: Literal["text", "latex"] = "text",
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
    Display the results of metrics computation.

    Args:
        metrics_result: result of metrics computation containing the metrics to display. If not provided,
            the result will be loaded from the checkpoint manager
        checkpoint_manager: if provided, will be used to load metrics result.
        table_metrics: metrics to tabulate. Entries can be :class:`~decent_bench.metrics.Metric` objects or strings
            (matching :attr:`~decent_bench.metrics.Metric.description`).
            If ``None`` all table metrics in metrics_result are displayed.
        plot_metrics: metrics to plot. Entries can be :class:`~decent_bench.metrics.Metric` objects or strings
            (matching :attr:`~decent_bench.metrics.Metric.description`).
            If ``None`` all plot metrics in metrics_result are displayed.
            If ``individual_plots`` is True, each metric is plotted in its own figure;
            otherwise a maximum of 3 metrics are plotted as subplots in the same figure.
        algorithms: algorithms to display. Entries can be :class:`~decent_bench.algorithms.Algorithm` objects or strings
            (matching :attr:`~decent_bench.algorithms.Algorithm.name`).
            If ``None`` all algorithms in metrics_result are displayed.
        table_fmt: table format, text is suitable for the terminal while latex can be copy-pasted into a latex document
        plot_grid: whether to show grid lines on the plots
        individual_plots: whether to plot each metric in a separate figure
        computational_cost: computational cost settings for plot metrics, if ``None`` x-axis will be iterations instead
            of computational cost
        scale_x_axis: scaling factor for computational cost x-axis, used to convert the cost units into more
            manageable units for plotting. Only used if ``computational_cost`` is provided.
        scale_compute: scaling factor for the compute-related metrics (i.e.
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
            if the checkpoint manager does not contain a valid metrics result to load.
        FileNotFoundError: If ``metrics_result`` is not provided and the checkpoint manager does not contain a metrics
            result file to load.

    Note:
        Checkpoint_manager is ignored if ``metrics_result`` is provided.

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
    selected_algorithms = _filter_algorithms(metrics_result, algorithms)
    selected_table_metrics = _filter_metrics(metrics_result, table_metrics, "table")
    selected_plot_metrics = _filter_metrics(metrics_result, plot_metrics, "plot")

    if not selected_algorithms:
        LOGGER.warning("No available algorithms were selected, stopping")
        return
    if not selected_table_metrics and not selected_plot_metrics:
        LOGGER.warning("No available metrics were selected, stopping")
        return

    # drop metrics that were not selected
    new_metrics_result = copy.deepcopy(metrics_result)

    if not selected_table_metrics:
        LOGGER.warning("No available table metrics were selected, skipping table")
        raw_table_results, table_results = None, None
    else:
        raw_table_results = _drop_metrics(selected_table_metrics, new_metrics_result.raw_table_results)
        table_results = _drop_df_rows(selected_table_metrics, new_metrics_result.table_results, "metric")

    if not selected_plot_metrics:
        LOGGER.warning("No available plot metrics were selected, skipping plots")
        raw_plot_results, plot_results = None, None
    else:
        raw_plot_results = _drop_metrics(selected_plot_metrics, new_metrics_result.raw_plot_results)
        plot_results = _drop_df_rows(selected_plot_metrics, new_metrics_result.plot_results, "metric")

    # drop algorithms that were not selected (from network_views and DataFrames)
    new_metrics_result.network_views = (
        None
        if not new_metrics_result.network_views
        else {
            alg: net_view
            for alg, net_view in new_metrics_result.network_views.items()
            if alg.name in selected_algorithms
        }
    )
    new_metrics_result.raw_table_results = _drop_algorithms(selected_algorithms, raw_table_results)
    new_metrics_result.table_results = _drop_df_rows(selected_algorithms, table_results, "algorithm")
    new_metrics_result.raw_plot_results = _drop_algorithms(selected_algorithms, raw_plot_results)
    new_metrics_result.plot_results = _drop_df_rows(selected_algorithms, plot_results, "algorithm")

    if save_path is not None:
        save_path = Path(save_path)
    elif save_path is None and checkpoint_manager is not None:
        save_path = checkpoint_manager.get_results_path()

    # display tables and/or plots
    if selected_table_metrics:
        display_tables(new_metrics_result, table_fmt=table_fmt, scale_compute=scale_compute, table_path=save_path)

    if selected_plot_metrics:
        display_plots(
            new_metrics_result,
            computational_cost=computational_cost,
            scale_x_axis=scale_x_axis,
            compare_iterations_and_computational_cost=compare_iterations_and_computational_cost,
            individual_plots=individual_plots,
            plot_grid=plot_grid,
            plot_format=plot_format,
            plot_path=save_path,
            show_plots=show_plots,
        )


def _filter_algorithms(metrics_result: MetricResult, algorithms: list["Algorithm[Network] | str"] | None) -> list[str]:
    if algorithms is None:
        return metrics_result.algorithms

    selected_algs = {_get_name_or_value(algorithm, "name") for algorithm in algorithms}
    available_algs = set(metrics_result.algorithms)
    missing_names = sorted(selected_algs - available_algs)
    if missing_names:
        LOGGER.warning(
            "Requested algorithm(s) were not found in metrics result, skipping: "
            f"{', '.join(missing_names)}. Available algorithms: {', '.join(metrics_result.algorithms)}"
        )

    return list(selected_algs & available_algs)


def _filter_metrics(
    metrics_result: MetricResult,
    metrics: list[Metric | str] | None,
    type_: Literal["table", "plot"],
) -> list[str]:
    available_metrics = metrics_result.table_metrics if type_ == "table" else metrics_result.plot_metrics
    if metrics is None:
        return available_metrics

    selected_metrics = {_get_name_or_value(metric, "description") for metric in metrics}
    set_available_metrics = set(available_metrics)
    missing_names = sorted(selected_metrics - set_available_metrics)
    if missing_names:
        LOGGER.warning(
            f"Requested {type_} metric(s) were not found in metrics result, skipping: "
            f"{', '.join(missing_names)}. Available metrics: {', '.join(set_available_metrics)}"
        )

    return list(selected_metrics & set_available_metrics)


def _get_name_or_value(value: object, attribute: str) -> str:
    return value if isinstance(value, str) else str(getattr(value, attribute))


def _drop_metrics(
    selected_metrics: list[str], frames: Mapping[Metric, pd.DataFrame] | None
) -> Mapping[Metric, pd.DataFrame] | None:
    if not frames:
        return None
    return {metric: frame for metric, frame in frames.items() if metric.description in selected_metrics}


def _drop_algorithms(
    selected_algs: list[str], frames: Mapping[Metric, pd.DataFrame] | None
) -> Mapping[Metric, pd.DataFrame] | None:
    if not frames:
        return None
    return {metric: frame[frame["algorithm"].isin(selected_algs)] for metric, frame in frames.items()}


def _drop_df_rows(selected: list[str], frame: pd.DataFrame | None, column: str) -> pd.DataFrame | None:
    if frame is None:
        return None
    return frame[frame[column].isin(selected)].copy()
