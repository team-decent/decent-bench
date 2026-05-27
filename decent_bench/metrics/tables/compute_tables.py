from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from decent_bench.utils.logger import LOGGER

if TYPE_CHECKING:
    from decent_bench.benchmark._metric_result import MetricResult

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


def validate_statistics_across_agents(statistics: list[str] | None) -> list[str]:
    """Validate and normalize requested statistic names."""
    # default to mean and std
    if statistics is None:
        return list(DEFAULT_STATISTICS.keys())

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
        return list(DEFAULT_STATISTICS.keys())

    return resolved_stats


def compute_tables(
    metric_result: "MetricResult",
    statistics_across_agents: list[str] | None,
) -> pd.DataFrame:
    """
    Aggregate table metrics data by mean+/-std across trials and statistics across agents.

    The resulting DataFrame is indexed by (metric, statistic, algorithm) and has columns mean and std.
    """
    stat_names = validate_statistics_across_agents(statistics_across_agents)
    statistics = {name: STATISTICS[name] for name in stat_names}
    rows: list[dict[str, object]] = []

    for metric, frame in metric_result.raw_table_results.items():

        for algorithm_name, algorithm_frame in frame.groupby(level="algorithm", sort=False):
            is_single_value_metric = algorithm_frame.index.get_level_values("agent").isna().all()
            statistics_to_use = {"": _single_value} if is_single_value_metric else statistics

            trial_values = [
                trial_series.astype(float).tolist()
                for _, trial_series in algorithm_frame.groupby(level="trial", sort=True)["value"]
            ]

            for stat_name, stat in statistics_to_use.items():
                agg_data_per_trial = [float(stat(values)) for values in trial_values]
                mean, std = _calculate_mean_and_std(agg_data_per_trial)
                rows.append(
                    {
                        "metric": metric.description,
                        "statistic": stat_name,
                        "algorithm": algorithm_name,
                        "mean": np.float32(mean),
                        "std": np.float32(std),
                    }
                )

    results = pd.DataFrame.from_records(rows, columns=["metric", "statistic", "algorithm", "mean", "std"])
    results = results.set_index(["metric", "statistic", "algorithm"]).sort_index()
    results[["mean", "std"]] = results[["mean", "std"]].astype("float32")
    metric_result.table_results = results[["mean", "std"]]
    return metric_result.table_results


def _single_value(values: Sequence[float]) -> float:
    return float(values[0])


def _calculate_mean_and_std(data: list[float]) -> tuple[float, float]:
    mean, std = np.mean(data), np.std(data)
    if np.isfinite(mean) and np.isfinite(std):
        return float(mean), float(std)
    return np.nan, np.nan
