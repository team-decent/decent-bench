import logging
from copy import deepcopy
from json import JSONDecodeError
from typing import TYPE_CHECKING, Literal

from rich.status import Status

from decent_bench.algorithms import Algorithm
from decent_bench.benchmark._benchmark_result import BenchmarkResult
from decent_bench.benchmark._metric_result import MetricResult
from decent_bench.benchmark.compute_metrics.compute_plots import (
    aggregate_plot_metrics,
    compute_plot_metrics,
)
from decent_bench.benchmark.compute_metrics.compute_tables import aggregate_table_metrics, compute_table_metrics
from decent_bench.metrics import Metric, utils
from decent_bench.metrics import metric_library as ml
from decent_bench.metrics._metrics_view import NetworkMetricsView
from decent_bench.networks import Network
from decent_bench.utils._metric_helpers import _find_duplicates
from decent_bench.utils.logger import LOGGER, start_logger

if TYPE_CHECKING:
    from decent_bench.benchmark import BenchmarkProblem

if TYPE_CHECKING:
    from decent_bench.benchmark import BenchmarkProblem
    from decent_bench.utils.checkpoint_manager import CheckpointManager


def compute_metrics(
    benchmark_result: BenchmarkResult | None = None,
    checkpoint_manager: "CheckpointManager | None" = None,
    *,
    table_metrics: list[Metric] | None = None,
    plot_metrics: list[Metric] | None = None,
    statistics_across_agents: list[str] | None = None,
    log_level: int = logging.INFO,
) -> MetricResult:
    """
    Compute metrics from a benchmark result.

    Args:
        benchmark_result: result of a benchmark execution. If not provided, the result will be loaded from the
            checkpoint manager
        checkpoint_manager: if provided, will be used to save results of metrics computation and/or load benchmark
            result.
        table_metrics: metrics to be displayed in a table of results. Table metrics are computed only at the
            *recentmost* iteration reached during benchmarking. If ``None``, all table
            metrics available for the benchmark problem will be used. For example, federated-only metrics are removed
            when a non-federated network is passed.
        plot_metrics: metrics to be plotted over algorithm iterations. Plot metrics are computed at *all* the
            iterations reached during benchmarking. If ``None``, all plot metrics available for the
            benchmark problem will be used.
        statistics_across_agents: statistics to compute across agents for metrics that return one value per agent
            (like ``ConsensusError`` or ``Accuracy``). Available statistics are "mean" (aliases "average", "avg"),
            "std", "max" (alias "maximum"), "min" (alias "minimum"), and "median" (alias "mdn"). If ``None``, "mean"
            and "std" are used.
        log_level: minimum level to log, e.g. :data:`logging.INFO`

    Returns:
        MetricsResult containing the computed metrics.

    Raises:
        ValueError: If neither ``benchmark_result`` nor ``checkpoint_manager`` is provided, or
            if the checkpoint manager does not contain a valid benchmark result to load.
        ValueError: If duplicate metrics (i.e. with same ``description``) are provided
            in ``table_metrics`` or ``plot_metrics``.

    Note:
        If ``benchmark_result`` is not provided, it will be loaded from the checkpoint manager. If both are provided,
        then the results from the provided ``benchmark_result`` will be used and the checkpoint manager will only be
        used to save the computed metrics result. If neither is provided, an error will be raised.

        All used table- and plot-metrics will be saved to the checkpoints' metadata if a checkpoint manager is provided,
        in order to know which metrics were computed and can be displayed later.

        Metrics that return ``False`` from :meth:`~decent_bench.metrics.Metric.is_available` for the given problem are
        filtered out from the returned metric lists. Warnings are emitted with the omitted metric names.

        Plot metrics can still be available even when their final table value is ``nan``: plot computation keeps the
        finite part of a trajectory, while table metrics are evaluated at the final iteration.

    """
    start_logger(log_level=log_level)
    LOGGER.info("Starting metrics computation")

    # 1) user input validation
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

    table_metrics, plot_metrics = _resolve_default_metrics(table_metrics, plot_metrics)

    # work on independent metric instances
    table_metrics = deepcopy(table_metrics)
    plot_metrics = deepcopy(plot_metrics)

    # check metrics are unique
    _validate_unique_descriptions(table_metrics, "table")
    _validate_unique_descriptions(plot_metrics, "plot")

    # remove unavailable metrics
    table_metrics = _remove_unavailable(table_metrics, benchmark_result.problem, "table")
    plot_metrics = _remove_unavailable(plot_metrics, benchmark_result.problem, "plot")


    # 2) compute table and plot metrics
    network_views: dict[Algorithm[Network], list[NetworkMetricsView]] = {}
    for alg, networks in benchmark_result.states.items():
        network_views[alg] = [NetworkMetricsView.from_network(nw) for nw in networks]

    iterations = _all_iterations(network_views)

    # compute metrics
    raw_plot_results = compute_plot_metrics(network_views, benchmark_result.problem, plot_metrics, iterations)
    raw_table_results = compute_table_metrics(network_views,
                                               benchmark_result.problem,
                                               table_metrics,
                                               iterations,
                                               raw_plot_results)

    # aggregate metrics
    aggregated_plot_metrics = aggregate_plot_metrics(raw_plot_results)
    aggregated_table_metrics = aggregate_table_metrics(raw_table_results, statistics_across_agents)

    # 2) create MetricResult
    result = MetricResult(
        network_views=network_views,
        raw_table_results=raw_table_results,
        raw_plot_results=raw_plot_results,
        table_results=aggregated_table_metrics,
        plot_results=aggregated_plot_metrics,
    )


    if checkpoint_manager is not None:
        with Status("Saving computed metrics..."):
            metadata = {
                "table_metrics": [metric.description for metric in table_metrics],
                "plot_metrics": [metric.description for metric in plot_metrics],
            }
            checkpoint_manager.save_metrics_result(result)
            checkpoint_manager.append_metadata(metadata)

    utils._clear_caches()  # noqa: SLF001

    return result


def _resolve_default_metrics(
    table_metrics: list[Metric] | None,
    plot_metrics: list[Metric] | None,
) -> tuple[list[Metric], list[Metric]]:
    if table_metrics is None:
        table_metrics = ml._DEFAULT_TABLE_METRICS  # noqa: SLF001
    if plot_metrics is None:
        plot_metrics = ml._DEFAULT_PLOT_METRICS  # noqa: SLF001
    return table_metrics, plot_metrics


def _validate_unique_descriptions(metrics: list[Metric], type_: Literal["table", "plot"]) -> None:
    duplicate_metric_descriptions = _find_duplicates([metric.description for metric in metrics])
    if duplicate_metric_descriptions:
        duplicates = ", ".join(duplicate_metric_descriptions)
        raise ValueError(f"{type_.capitalize()} metric descriptions must be unique, duplicates found: {duplicates}")


def _remove_unavailable(
    metrics: list[Metric], problem: "BenchmarkProblem", type_: Literal["table", "plot"]
) -> list[Metric]:
    available_metrics: list[Metric] = []
    for metric in metrics:
        available, reason = metric.is_available(problem)
        if not available:
            LOGGER.warning(f"Skipping {type_} metric '{metric.description}' because it is unavailable: {reason}")
            continue

        available_metrics.append(metric)

    return available_metrics

def _all_iterations(network_views: dict[Algorithm[Network], list[NetworkMetricsView]]) -> list[int]:
    """Find all the iterations that were reached in at least one trial by at least one algorithm."""
    iterations: list[int] = []
    for network_views_by_trial in network_views.values():
        for network_view in network_views_by_trial:
            iterations += network_view.iterations

    return list(set(iterations))
