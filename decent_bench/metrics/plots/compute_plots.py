from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from decent_bench.utils.logger import LOGGER

if TYPE_CHECKING:
    from decent_bench.benchmark._metric_result import MetricResult

# upper bound to values that can be plotted on the y-axis (both in linear and log scale)
# values above this threshold will be removed
MAX_Y_PLOT_VALUE = 1e30


def compute_plots(
    metric_result: "MetricResult",
) -> pd.DataFrame:
    """
    Aggregate plot metrics data by mean, min, and max across trials.

    The resulting DataFrame is indexed by (metric, algorithm, iterations) and has columns
    (y_mean, y_min, y_max).
    """
    rows: list[dict[str, object]] = []

    for metric, frame in metric_result.raw_plot_results.items():

        for algorithm_name, algorithm_frame in frame.groupby(level="algorithm", sort=False):
            trial_series: list[pd.Series] = []
            had_non_finite = False

            for _, per_trial_frame in algorithm_frame.groupby(level="trial", sort=True):
                by_iter = per_trial_frame.groupby(level="iterations")["value"].mean().sort_index()
                values = by_iter.to_numpy(dtype=float)
                invalid = ~np.isfinite(values)
                if invalid.any():
                    had_non_finite = True
                    first_invalid = int(np.argmax(invalid))
                    by_iter = by_iter.iloc[:first_invalid]

                if len(by_iter) < len(values):
                    had_non_finite = True

                trial_series.append(by_iter.astype("float32"))

            if not trial_series:
                if had_non_finite:
                    LOGGER.warning(
                        f"Skipping plot computation for {metric.description} and {algorithm_name}: "
                        "all trials diverged before the first plottable datapoint."
                    )
                else:
                    LOGGER.warning(
                        f"Skipping plot computation for {metric.description} and {algorithm_name}: "
                        "metric produced no datapoints."
                    )
                continue

            common_prefix_length = min(len(series) for series in trial_series)
            aligned = [series.iloc[:common_prefix_length] for series in trial_series]

            if had_non_finite:
                LOGGER.info(
                    f"Truncating plot computation for {metric.description} and {algorithm_name} "
                    "at the first non-finite or over-threshold datapoint; retained "
                    f"{common_prefix_length} point(s) from {len(aligned)}/{len(trial_series)} trial(s)."
                )

            x_values = aligned[0].index.to_list()
            y_matrix = np.vstack([series.to_numpy(dtype=float) for series in aligned])
            y_mean = y_matrix.mean(axis=0)
            y_min = y_matrix.min(axis=0)
            y_max = y_matrix.max(axis=0)

            for x_val, y_m, y_lo, y_hi in zip(x_values, y_mean, y_min, y_max, strict=True):
                rows.append(
                    {
                        "metric": metric.description,
                        "algorithm": str(algorithm_name),
                        "iterations": x_val,
                        "y_mean": y_m,
                        "y_min": y_lo,
                        "y_max": y_hi,
                    }
                )

    results = pd.DataFrame.from_records(
        rows,
        columns=["metric", "algorithm", "iterations", "y_mean", "y_min", "y_max"],
    )

    results[["y_mean", "y_min", "y_max"]] = results[["y_mean", "y_min", "y_max"]].astype("float32")
    return results.set_index(["metric", "algorithm", "iterations"]).sort_index()[["y_mean", "y_min", "y_max"]]
