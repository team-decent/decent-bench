import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import tabulate as tb
from scipy import stats

import decent_bench.metrics.metric_utils as utils
from decent_bench.algorithms import Algorithm
from decent_bench.costs import EmpiricalRiskCost
from decent_bench.metrics._metric import Metric
from decent_bench.metrics._metrics_view import NetworkMetricsView
from decent_bench.networks import Network
from decent_bench.utils.logger import LOGGER

from .metric_library import (
    FunctionCalls,
    GradientCalls,
    HessianCalls,
    ProximalCalls,
)

if TYPE_CHECKING:
    from decent_bench.benchmark import BenchmarkProblem, MetricResult

STATISTICS_ABBR = {"average": "avg", "median": "mdn"}
SCALE_METRICS = (FunctionCalls, GradientCalls, HessianCalls, ProximalCalls)


def display_tables(
    metrics_result: "MetricResult",
    table_fmt: Literal["grid", "latex"] = "grid",
    scale_compute: float = 1.0,
    table_path: Path | None = None,
) -> None:
    """
    Display table of metrics as mean+/-std across trials, and statistics across agents, one column per algorithm.

    Args:
        metrics_result: result of metrics computation containing the metrics to display.
        table_fmt: table format, grid is suitable for the terminal while latex can be copy-pasted into a latex document.
        scale_compute: scaling factor for the compute related metrics (i.e.
            :class:`~decent_bench.metrics.metric_library.FunctionCalls`,
            :class:`~decent_bench.metrics.metric_library.GradientCalls`,
            :class:`~decent_bench.metrics.metric_library.HessianCalls` and
            :class:`~decent_bench.metrics.metric_library.ProximalCalls`) shown in the table, used to convert the
            raw count into more manageable units for display.
        table_path: optional directory path to save the tables in. Tables will be saved as "table.txt" and "table.tex".
            If not provided, the tables will only be displayed.

    """
    if metrics_result.table_metrics is None or metrics_result.table_results is None:
        return

    if (
        any(isinstance(metric, SCALE_METRICS) for metric in metrics_result.table_metrics)
        and metrics_result.network_views
    ):
        network_view = next(iter(metrics_result.network_views.values()))[0]
        metric_views = network_view.agents()
        scale_metrics_in_use = [
            metric.description for metric in metrics_result.table_metrics if isinstance(metric, SCALE_METRICS)
        ]
        if any(isinstance(a.cost, EmpiricalRiskCost) for a in metric_views):
            LOGGER.info(
                f"Empirical-risk cost functions are in use. Compute counters increment by the number of samples "
                f"processed in each method call, which can lead to large raw counts. Applying scaling factor of "
                f"'scale_compute={scale_compute}' to {scale_metrics_in_use} metrics for display."
            )

    table_results = metrics_result.table_results
    algs = list(table_results.keys())
    headers = ["Metric (statistic across agents)"] + [alg.name for alg in algs]
    rows: list[list[str]] = []

    for metric in metrics_result.table_metrics:
        for statistic_name in table_results[algs[0]][metric]:
            row = [f"{metric.description} ({statistic_name})"] if statistic_name else [f"{metric.description}"]
            for alg in algs:
                mean, std = table_results[alg][metric][statistic_name]

                if isinstance(metric, SCALE_METRICS):
                    mean, std = mean * scale_compute, std * scale_compute

                formatted_mean_std = _format_mean_std(
                    mean,
                    std,
                    metric.fmt,
                )
                row.append(formatted_mean_std)
            rows.append(row)

    grid_table = tb.tabulate(rows, headers, tablefmt="grid")
    latex_table = tb.tabulate(rows, headers, tablefmt="latex")
    LOGGER.info("\n" + latex_table if table_fmt == "latex" else "\n" + grid_table)

    if table_path:
        # Save both latex and grid tables to the specified directory
        table_path.mkdir(parents=True, exist_ok=True)
        latex_path = table_path / "table.tex"
        grid_path = table_path / "table.txt"
        latex_path.write_text(latex_table, encoding="utf-8")
        grid_path.write_text(grid_table, encoding="utf-8")
        LOGGER.info(f"Saved LaTeX table to {latex_path}")
        LOGGER.info(f"Saved grid table to {grid_path}")


def compute_tables(
    resulting_network_views: dict[Algorithm[Network], list[NetworkMetricsView]],
    problem: "BenchmarkProblem",
    metrics: list[Metric],
) -> Mapping[Algorithm[Network], Mapping[Metric, Mapping[str, tuple[float, float]]]]:
    """
    Compute table metrics with mean and std across trials.

    Args:
        resulting_network_views: resulting network views from the trial executions, grouped by algorithm
        problem: benchmark problem whose properties are used for metric calculations
        metrics: metrics to calculate

    Returns:
        A nested dictionary containing the mean and std for each metric and statistic, structured as
        {algorithm: {metric: {statistic_name: (mean, std)}}}

    """
    if not metrics:
        return {}

    algs = list(resulting_network_views)
    results: dict[Algorithm[Network], dict[Metric, dict[str, tuple[float, float]]]] = {a: {} for a in algs}

    with warnings.catch_warnings(action="ignore"), utils.MetricProgressBar() as progress:
        table_task = progress.add_task(
            "Computing table metrics",
            total=sum(len(metric.statistics) for metric in metrics),
            status="",
        )
        for metric in metrics:
            progress.update(table_task, status=f"Task: {metric.description}")

            data_per_trial = [
                _table_data_per_trial(
                    resulting_network_views[a],
                    problem,
                    metric,
                )
                for a in algs
            ]

            for statistic in metric.statistics:
                statistic_name = STATISTICS_ABBR.get(statistic.__name__) or statistic.__name__
                if statistic_name == "default_statistic":
                    statistic_name = ""
                for i, alg in enumerate(algs):
                    agg_data_per_trial = [statistic(trial) for trial in data_per_trial[i]]
                    mean, std = _calculate_mean_and_std(agg_data_per_trial)

                    if metric not in results[alg]:
                        results[alg][metric] = {}
                    results[alg][metric][statistic_name] = (mean, std)

                progress.advance(table_task)
        progress.update(table_task, status="Table computation complete")

    return results


def _table_data_per_trial(
    network_views_per_trial: list[NetworkMetricsView],
    problem: "BenchmarkProblem",
    metric: Metric,
) -> list[Sequence[float]]:
    return [metric.get_table_data(network_view, problem) for network_view in network_views_per_trial]


def _calculate_mean_and_std(data: list[float]) -> tuple[float, float]:
    mean, std = np.mean(data), np.std(data)
    if np.isfinite(mean) and np.isfinite(std):
        return (float(mean), float(std))

    return np.nan, np.nan


def _format_mean_std(mean: float, std: float, fmt: str) -> str:
    if not _is_valid_float_format_spec(fmt):
        LOGGER.warning(f"Invalid format string '{fmt}', defaulting to scientific notation")
        fmt = ".2e"

    return f"{mean:{fmt}} \u00b1 {std:{fmt}}"


def _is_valid_float_format_spec(fmt: str) -> bool:
    """
    Validate that the given format spec can be used to format a float.

    This avoids attempting to format real values with an invalid format string.

    """
    try:
        f"{0.01:{fmt}}"
    except (ValueError, TypeError):
        return False
    return True
