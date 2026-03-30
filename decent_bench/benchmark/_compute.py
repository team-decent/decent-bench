import logging
from collections.abc import Mapping, Sequence
from copy import deepcopy
from json import JSONDecodeError
from typing import TYPE_CHECKING, cast

from decent_bench.agents import AgentMetricsView
from decent_bench.benchmark._benchmark_result import BenchmarkResult
from decent_bench.benchmark._metric_result import MetricResult
from decent_bench.distributed_algorithms import Algorithm
from decent_bench.metrics import Metric, compute_plots, compute_tables
from decent_bench.metrics import metric_library as ml
from decent_bench.networks import Network
from decent_bench.utils import logger

if TYPE_CHECKING:
    from decent_bench.utils.checkpoint_manager import CheckpointManager


PlotMetricData = tuple[Sequence[float], Sequence[float], Sequence[float], Sequence[float]]


def compute_metrics(
    benchmark_result: BenchmarkResult | None = None,
    checkpoint_manager: "CheckpointManager | None" = None,
    *,
    table_metrics: list[Metric] = ml.DEFAULT_TABLE_METRICS,
    plot_metrics: list[Metric] | list[list[Metric]] = ml.DEFAULT_PLOT_METRICS,
    confidence_level: float = 0.95,
    log_level: int = logging.INFO,
) -> MetricResult:
    """
    Compute metrics from a benchmark result.

    Args:
        benchmark_result: result of a benchmark execution. If not provided, the result will be loaded from the
            checkpoint manager
        checkpoint_manager: if provided, will be used to save results of metrics computation and/or load benchmark
            result.
        table_metrics: metrics to tabulate as confidence intervals after the execution, defaults to
            :const:`~decent_bench.metrics.metric_library.DEFAULT_TABLE_METRICS`
        plot_metrics: metrics to plot after the execution, defaults to
            :const:`~decent_bench.metrics.metric_library.DEFAULT_PLOT_METRICS`.
            If a list of lists is provided, each inner list will be plotted in a separate figure. Otherwise up to 3
            metrics will be grouped and plotted in their own figure with subplots.
        confidence_level: confidence level for computing confidence intervals of the table metrics, expressed as a value
            between 0 and 1 (e.g., 0.95 for 95% confidence, 0.99 for 99% confidence). Higher values result in
            wider confidence intervals.
        log_level: minimum level to log, e.g. :data:`logging.INFO`

    Returns:
        MetricsResult containing the computed metrics.

    Raises:
        ValueError: If neither ``benchmark_result`` nor ``checkpoint_manager`` is provided, or
            if the checkpoint manager does not contain a valid benchmark result to load.

    Note:
        If ``benchmark_result`` is not provided, it will be loaded from the checkpoint manager. If both are provided,
        then the results from the provided ``benchmark_result`` will be used and the checkpoint manager will only be
        used to save the computed metrics result. If neither is provided, an error will be raised.

        All used table- and plot-metrics will be saved to the checkpoints' metadata if a checkpoint manager is provided,
        in order to know which metrics were computed and can be displayed later.

        Metrics that mark themselves unavailable during computation (through ``Metric.mark_unavailable``), are
        filtered out from the returned metric lists. Warnings are emitted with the omitted metric names.

        Plot metrics can still be available even when their final table value is ``nan``: plot computation keeps the
        finite part of a trajectory, while table metrics are evaluated at the final iteration.

    """
    logger.start_logger(log_level=log_level)

    if benchmark_result is None:
        if checkpoint_manager is None:
            raise ValueError(
                "If ``benchmark_result`` is not provided, ``checkpoint_manager`` must be provided "
                "to load the benchmark result from."
            )

        try:
            benchmark_result = checkpoint_manager.load_benchmark_result()
        except (FileNotFoundError, KeyError) as e:
            raise ValueError(f"Invalid checkpoint directory: missing or corrupted metadata - {e}") from e
        except JSONDecodeError as e:
            raise ValueError(f"Invalid checkpoint directory: metadata is not valid JSON - {e}") from e

        if len(benchmark_result.states) == 0:
            raise ValueError("No benchmark result found in checkpoint manager to compute metrics")

    # work on independent metric instances so any metric state does not leak globally
    table_metrics = deepcopy(table_metrics)
    plot_metrics = deepcopy(plot_metrics)

    resulting_agent_states: dict[Algorithm[Network], list[list[AgentMetricsView]]] = {}
    for alg, networks in benchmark_result.states.items():
        resulting_agent_states[alg] = [[AgentMetricsView.from_agent(a) for a in nw.agents()] for nw in networks]
    table_results = compute_tables(resulting_agent_states, benchmark_result.problem, table_metrics, confidence_level)
    plot_results = compute_plots(resulting_agent_states, benchmark_result.problem, plot_metrics)

    table_metrics = _filter_table_metrics_with_results(table_metrics, table_results)
    plot_metrics = _filter_plot_metrics_with_results(plot_metrics, plot_results)

    result = MetricResult(
        agent_metrics=resulting_agent_states,
        table_metrics=table_metrics,
        plot_metrics=plot_metrics,
        table_results=table_results,
        plot_results=plot_results,
    )

    if checkpoint_manager is not None:
        if any(isinstance(m, list) for m in plot_metrics):
            grouped_plot_metrics = cast("list[list[Metric]]", plot_metrics)
            flat_metrics = [metric for group in grouped_plot_metrics for metric in group]
        else:
            flat_metrics = cast("list[Metric]", plot_metrics)
        metadata = {
            "table_metrics": [metric.table_description for metric in table_metrics],
            "plot_metrics": [metric.plot_description for metric in flat_metrics],
        }
        checkpoint_manager.save_metrics_result(result)
        checkpoint_manager.append_metadata(metadata)

    return result


def _filter_table_metrics_with_results(
    table_metrics: list[Metric],
    table_results: Mapping[Algorithm[Network], Mapping[Metric, Mapping[str, tuple[float, float]]]],
) -> list[Metric]:
    filtered_metrics = [
        metric for metric in table_metrics if any(metric in results for results in table_results.values())
    ]
    omitted = [metric for metric in table_metrics if metric not in filtered_metrics]
    if omitted:
        logger.LOGGER.warning(
            f"Omitting unavailable table metrics: {', '.join(metric.table_description for metric in omitted)}"
        )
    return filtered_metrics


def _filter_plot_metrics_with_results(
    plot_metrics: list[Metric] | list[list[Metric]],
    plot_results: Mapping[Algorithm[Network], Mapping[Metric, PlotMetricData]],
) -> list[Metric] | list[list[Metric]]:
    available_metrics = {metric for results in plot_results.values() for metric in results}

    if any(isinstance(metric, list) for metric in plot_metrics):
        grouped_plot_metrics = cast("list[list[Metric]]", plot_metrics)
        filtered_groups: list[list[Metric]] = []
        omitted: list[Metric] = []
        for group in grouped_plot_metrics:
            filtered_group = [metric for metric in group if metric in available_metrics]
            omitted.extend(metric for metric in group if metric not in available_metrics)
            if filtered_group:
                filtered_groups.append(filtered_group)
        if omitted:
            logger.LOGGER.warning(
                f"Omitting unavailable plot metrics: {', '.join(metric.plot_description for metric in omitted)}"
            )
        return filtered_groups

    flat_plot_metrics = cast("list[Metric]", plot_metrics)
    filtered_metrics = [metric for metric in flat_plot_metrics if metric in available_metrics]
    omitted = [metric for metric in flat_plot_metrics if metric not in available_metrics]
    if omitted:
        logger.LOGGER.warning(
            f"Omitting unavailable plot metrics: {', '.join(metric.plot_description for metric in omitted)}"
        )
    return filtered_metrics
