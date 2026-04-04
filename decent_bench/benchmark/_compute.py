import logging
from copy import deepcopy
from json import JSONDecodeError
from typing import TYPE_CHECKING, Literal, cast

from decent_bench.agents import AgentMetricsView
from decent_bench.benchmark._benchmark_result import BenchmarkResult
from decent_bench.benchmark._metric_result import MetricResult
from decent_bench.distributed_algorithms import Algorithm
from decent_bench.metrics import Metric, compute_plots, compute_tables
from decent_bench.metrics import metric_library as ml
from decent_bench.networks import Network
from decent_bench.utils._metric_helpers import _find_duplicates, _flatten_plot_metrics
from decent_bench.utils.logger import LOGGER, start_logger

if TYPE_CHECKING:
    from decent_bench.benchmark import BenchmarkProblem
    from decent_bench.utils.checkpoint_manager import CheckpointManager


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
        ValueError: If duplicate metrics (i.e. with same ``table_description`` or ``plot_description``) are provided
            in ``table_metrics`` or ``plot_metrics``.

    Note:
        If ``benchmark_result`` is not provided, it will be loaded from the checkpoint manager. If both are provided,
        then the results from the provided ``benchmark_result`` will be used and the checkpoint manager will only be
        used to save the computed metrics result. If neither is provided, an error will be raised.

        All used table- and plot-metrics will be saved to the checkpoints' metadata if a checkpoint manager is provided,
        in order to know which metrics were computed and can be displayed later.

        Metrics that return ``False`` from :meth:`~decent_bench.metrics.Metric.is_available` for the given problem
        filtered out from the returned metric lists. Warnings are emitted with the omitted metric names.

        Plot metrics can still be available even when their final table value is ``nan``: plot computation keeps the
        finite part of a trajectory, while table metrics are evaluated at the final iteration.

    """
    start_logger(log_level=log_level)

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

    # work on independent metric instances
    table_metrics = deepcopy(table_metrics)
    plot_metrics = deepcopy(plot_metrics)

    # check metrics are unique, which also checks that plot_metrics is either list or list of lists
    _validate_unique_metric_descriptions(table_metrics, plot_metrics)

    # remove unavailable tables
    table_metrics = _remove_unavailable(table_metrics, benchmark_result.problem, "table")

    if any(isinstance(metric, list) for metric in plot_metrics):
        grouped_plot_metrics = cast("list[list[Metric]]", plot_metrics)
        for i, group in enumerate(grouped_plot_metrics):
            grouped_plot_metrics[i] = _remove_unavailable(group, benchmark_result.problem, "plot")
        plot_metrics = grouped_plot_metrics
    else:
        plot_metrics = _remove_unavailable(cast("list[Metric]", plot_metrics), benchmark_result.problem, "plot")

    # compute table and plot results
    resulting_agent_states: dict[Algorithm[Network], list[list[AgentMetricsView]]] = {}
    for alg, networks in benchmark_result.states.items():
        resulting_agent_states[alg] = [[AgentMetricsView.from_agent(a) for a in nw.agents()] for nw in networks]
    table_results = compute_tables(resulting_agent_states, benchmark_result.problem, table_metrics, confidence_level)
    plot_results = compute_plots(resulting_agent_states, benchmark_result.problem, plot_metrics)

    result = MetricResult(
        agent_metrics=resulting_agent_states,
        table_metrics=table_metrics,
        plot_metrics=plot_metrics,
        table_results=table_results,
        plot_results=plot_results,
    )

    if checkpoint_manager is not None:
        flat_metrics = _flatten_plot_metrics(plot_metrics)
        metadata = {
            "table_metrics": [metric.table_description for metric in table_metrics],
            "plot_metrics": [metric.plot_description for metric in flat_metrics],
        }
        checkpoint_manager.save_metrics_result(result)
        checkpoint_manager.append_metadata(metadata)

    return result


def _validate_unique_metric_descriptions(
    table_metrics: list[Metric],
    plot_metrics: list[Metric] | list[list[Metric]],
) -> None:
    duplicate_table_descriptions = _find_duplicates([metric.table_description for metric in table_metrics])
    if duplicate_table_descriptions:
        duplicates = ", ".join(duplicate_table_descriptions)
        raise ValueError(f"Table metric descriptions must be unique, duplicates found: {duplicates}")

    duplicate_plot_descriptions = _find_duplicates([
        metric.plot_description for metric in _flatten_plot_metrics(plot_metrics)
    ])
    if duplicate_plot_descriptions:
        duplicates = ", ".join(duplicate_plot_descriptions)
        raise ValueError(f"Plot metric descriptions must be unique, duplicates found: {duplicates}")


def _remove_unavailable(
    metrics: list[Metric], problem: "BenchmarkProblem", type_: Literal["table", "plot"]
) -> list[Metric]:
    available_metrics: list[Metric] = []
    for metric in metrics:
        available, reason = metric.is_available(problem)
        if not available:
            description = metric.table_description if type_ == "table" else metric.plot_description
            LOGGER.warning(f"Skipping {type_} metric '{description}' because it is unavailable: {reason}")
            continue

        available_metrics.append(metric)

    return available_metrics
