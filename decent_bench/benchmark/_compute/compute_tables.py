from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from decent_bench.algorithms import Algorithm
from decent_bench.benchmark._compute.compute_metrics_at_iter import compute_metrics_at_iter
from decent_bench.metrics import Metric, utils
from decent_bench.metrics._metrics_view import NetworkMetricsView
from decent_bench.networks import Network
from decent_bench.utils.logger import LOGGER

if TYPE_CHECKING:
    from decent_bench.benchmark import BenchmarkProblem

if TYPE_CHECKING:
    from decent_bench.benchmark import BenchmarkProblem


STATISTICS: dict[str, Callable[[Sequence[float]], float]] = {
    "mean": np.mean,
    "std": np.std,
    "max": np.max,
    "min": np.min,
    "median": np.median,
}
STATISTICS_ALIASES = {
    "average": "mean",
    "avg": "mean",
    "maximum": "max",
    "minimum": "min",
    "mdn": "median",
}
DEFAULT_STATISTICS = {"mean": STATISTICS["mean"], "std": STATISTICS["std"]}


def _resolve_statistics(statistics: list[str] | None) -> dict[str, Callable[[Sequence[float]], float]]:
    """Resolve statistics, defaulting to mean and std if they cannot be resolved."""
    # default to mean and std
    if statistics is None:
        return DEFAULT_STATISTICS

    resolved_stats: list[str] = []
    for stat in statistics:
        resolved_stat = STATISTICS_ALIASES.get(stat, stat)
        if resolved_stat not in STATISTICS:
            LOGGER.warning(f"Skipping {stat} because it is not a valid statistic (or alias)")
            continue
        resolved_stats.append(resolved_stat)

    if not resolved_stats:
        LOGGER.warning(
            f"No valid statistic was passed, defaulting to mean and std; "
            f"valid stats: {', '.join(STATISTICS.keys())}, passed: {', '.join(statistics)}"
        )
        return DEFAULT_STATISTICS

    return {name: STATISTICS[name] for name in resolved_stats}


def compute_table_metrics(
    network_views: dict[Algorithm[Network], list[NetworkMetricsView]],
    problem: "BenchmarkProblem",
    metrics: list[Metric],
    iterations: list[int],
    plot_metrics_results: Mapping[Metric, pd.DataFrame] | None = None,
) -> Mapping[Metric, pd.DataFrame]:
    """
    Compute metrics at the final iteration and return one DataFrame per metric.

    If ``plot_metrics_results`` is not None and contains a metric from *metrics*, the DataFrame is extracted from
    ``plot_metrics_results`` instead of recomputing it.
    The DataFrame has columns (algorithm, trial, agent, value), since the iteration column is dropped.
    """
    if not plot_metrics_results:
        plot_metrics_results = {}
    frames_by_metric: dict[Metric, pd.DataFrame] = {}
    already_computed: set[Metric] = set() if not plot_metrics_results else set(metrics) & set(plot_metrics_results)
    if not iterations:
        return frames_by_metric
    final_iteration = max(iterations)

    with utils.MetricProgressBar() as progress:
        table_task = progress.add_task("Computing table metrics", total=len(metrics), status="")

        for metric in metrics:
            progress.update(table_task, status=f"Task: {metric.description}")

            if metric in already_computed:
                plot_frame = plot_metrics_results[metric]
                frames_by_metric[metric] = plot_frame.loc[plot_frame["iteration"] == final_iteration]
            else:
                frames_by_metric[metric] = compute_metrics_at_iter(network_views, problem, metric, final_iteration)
            progress.advance(table_task)

            frames_by_metric[metric] = frames_by_metric[metric].drop("iteration", axis="columns")

        progress.update(table_task, status="Table computation complete")

    return frames_by_metric


def aggregate_table_metrics(
    table_results: Mapping[Metric, pd.DataFrame],
    statistics: list[str] | None = None,
) -> pd.DataFrame | None:
    """
    Aggregate table metrics by statistics across agents and by mean, std across trials.

    The DataFrame that is returned has columns (metric, algorithm, statistic, mean, std). Numerical values are cast
    to float32.
    """
    resolved_statistics = _resolve_statistics(statistics)
    frames_by_metric: list[pd.DataFrame] = []

    for metric, frame in table_results.items():
        # if there is a single value (agent is always 0), drop agent column and add a dummy statistic column
        if len(frame["agent"].unique()) == 1:
            new_frame = frame.loc[:, ["algorithm", "trial", "value"]].copy()
            new_frame["statistic"] = ""
            new_frame = new_frame[["algorithm", "trial", "statistic", "value"]]  # reorder columns

        # if there are per-agent values, compute statistics across them
        # gives DataFrame with columns (algorithm, trial, statistic, value)
        else:
            # compute statistics
            agg_spec = {name: ("value", func) for name, func in resolved_statistics.items()}
            new_frame = frame.groupby(["algorithm", "trial"]).agg(**agg_spec).reset_index()
            # turn the statistics into a new column
            new_frame = new_frame.melt(
                id_vars=["algorithm", "trial"],
                value_vars=list(resolved_statistics.keys()),
                var_name="statistic",
                value_name="value",
            )

        # 3) compute mean and std across trials, gives DataFrame with (algorithm, statistic, mean, std)
        new_frame = (
            new_frame.groupby(["algorithm", "statistic"])["value"]
            .agg(mean="mean", std=lambda x: x.std(ddof=0))
            .reset_index()
        )

        # 4) add metric column
        new_frame = new_frame.assign(metric=metric.description)
        new_frame = new_frame[["metric", "algorithm", "statistic", "mean", "std"]]  # reorder columns
        new_frame[["mean", "std"]] = new_frame[["mean", "std"]].astype("float32")

        frames_by_metric.append(new_frame)

    # 5) return concatenated frames
    if not frames_by_metric:
        return None
    return pd.concat(frames_by_metric, ignore_index=True)
