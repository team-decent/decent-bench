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
from decent_bench.utils._metric_helpers import _flatten_plot_metrics
from decent_bench.utils.logger import LOGGER

if TYPE_CHECKING:
    from decent_bench.distributed_algorithms import Algorithm
    from decent_bench.networks import Network
    from decent_bench.utils.checkpoint_manager import CheckpointManager


def display_metrics(
    metrics_result: MetricResult | None = None,
    checkpoint_manager: "CheckpointManager | None" = None,
    *,
    table_metrics: list[Metric | str] | None = None,
    plot_metrics: list[Metric | str] | list[list[Metric | str]] | None = None,
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
            (matching :attr:`~decent_bench.metrics.Metric.table_description`).
        plot_metrics: metrics to plot, defaults to ``None`` which will display all metrics in the metrics_result.
            Entries can be :class:`~decent_bench.metrics.Metric` objects or strings
            (matching :attr:`~decent_bench.metrics.Metric.plot_description`).
            If a list of lists is provided, each inner list will be plotted in a separate figure. Otherwise up to 3
            metrics will be grouped and plotted in their own figure with subplots.
        algorithms: algorithms to display. If provided, only these algorithms are included in tables and plots.
            Entries can be :class:`~decent_bench.distributed_algorithms.Algorithm` objects or strings
            (matching :attr:`~decent_bench.distributed_algorithms.Algorithm.name`).
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

    # filter `metrics_result` based on `plot_metrics`, `table_metrics`, and `algorithms` (if provided)
    new_metrics_result = deepcopy(metrics_result)

    if table_metrics is not None:
        new_metrics_result.table_metrics = _get_new_table_metrics(new_metrics_result, table_metrics)

    if plot_metrics is not None:
        new_metrics_result.plot_metrics = _get_new_plot_metrics(new_metrics_result, plot_metrics)

    if algorithms is not None:
        new_metrics_result = _filter_algorithms(new_metrics_result, algorithms)

    # check that filtering didn't empty out the displayable results
    if algorithms is not None and not new_metrics_result.available_algorithms:
        raise ValueError(
            f"No algorithms remain after filtering. Requested algorithms not found in metrics result. "
            f"Available algorithms: {', '.join(metrics_result.available_algorithms)}"
        )

    if table_metrics is not None and plot_metrics is not None:
        if not new_metrics_result.available_table_metrics and not new_metrics_result.available_plot_metrics:
            raise ValueError(
                f"No table or plot metrics remain after filtering. "
                f"Available table metrics: {', '.join(metrics_result.available_table_metrics)}. "
                f"Available plot metrics: {', '.join(metrics_result.available_plot_metrics)}"
            )

    if save_path is not None:
        save_path = Path(save_path)
    elif save_path is None and checkpoint_manager is not None:
        save_path = checkpoint_manager.get_results_path()

    if new_metrics_result.table_metrics:
        display_tables(new_metrics_result, table_fmt=table_fmt, scale_compute=scale_compute, table_path=save_path)
    else:
        LOGGER.warning("No table metrics to display, skipping tables")

    if new_metrics_result.plot_metrics:
        display_plots(
            new_metrics_result,
            computational_cost=computational_cost,
            scale_x_axis=scale_x_axis,
            compare_iterations_and_computational_cost=compare_iterations_and_computational_cost,
            individual_plots=individual_plots,
            plot_grid=plot_grid,
            plot_format=plot_format,
            plot_path=save_path,
        )
    else:
        LOGGER.warning("No plot metrics to display, skipping plots")


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

    for attribute in ("agent_metrics", "table_results", "plot_results"):
        mapping = getattr(metrics_result, attribute)
        if mapping is not None:
            setattr(
                metrics_result,
                attribute,
                {algorithm: values for algorithm, values in mapping.items() if algorithm.name in selected_names},
            )

    return metrics_result


def _get_new_table_metrics(
    metrics_result: MetricResult,
    table_metrics: list[Metric | str],
) -> list[Metric]:
    if metrics_result.table_metrics is None:
        return []

    table_lookup = _build_metric_lookup(metrics_result.table_metrics, description_type="table")

    new_table_metrics = []
    for metric_or_name in table_metrics:
        metric_name = _get_name_or_value(metric_or_name, "table_description")
        original_metric = table_lookup.get(metric_name)
        if original_metric is None:
            LOGGER.warning(f"Requested table metric '{metric_name}' not found in metrics result, skipping")
            continue
        new_table_metrics.append(original_metric)

    return new_table_metrics


def _get_new_plot_metrics(
    metrics_result: MetricResult,
    plot_metrics: list[Metric | str] | list[list[Metric | str]],
) -> list[Metric] | list[list[Metric]]:
    if metrics_result.plot_metrics is None:
        return []

    flat_metrics = _flatten_plot_metrics(metrics_result.plot_metrics)
    plot_lookup = _build_metric_lookup(flat_metrics, description_type="plot")

    new_plot_metrics = []
    for group in plot_metrics:
        if isinstance(group, list):
            new_group = []
            for metric_or_name in group:
                metric_name = _get_name_or_value(metric_or_name, "plot_description")
                original_metric = plot_lookup.get(metric_name)
                if original_metric is None:
                    LOGGER.warning(f"Requested plot metric '{metric_name}' not found in metrics result, skipping")
                    continue
                new_group.append(original_metric)
            if new_group:
                new_plot_metrics.append(new_group)
        else:
            metric_name = _get_name_or_value(group, "plot_description")
            original_metric = plot_lookup.get(metric_name)
            if original_metric is None:
                LOGGER.warning(f"Requested plot metric '{metric_name}' not found in metrics result, skipping")
                continue
            new_plot_metrics.append(original_metric)  # type: ignore[arg-type]

    return new_plot_metrics


def _build_metric_lookup(metrics: list[Metric], *, description_type: Literal["table", "plot"]) -> dict[str, Metric]:
    lookup: dict[str, Metric] = {}
    for metric in metrics:
        description = metric.table_description if description_type == "table" else metric.plot_description
        lookup.setdefault(description, metric)

    return lookup


def _get_name_or_value(value: object, attribute: str) -> str:
    return value if isinstance(value, str) else str(getattr(value, attribute))
