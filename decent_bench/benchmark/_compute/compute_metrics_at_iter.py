from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from decent_bench.algorithms import Algorithm
from decent_bench.metrics._metric import Metric
from decent_bench.metrics._metrics_view import NetworkMetricsView
from decent_bench.networks import Network

if TYPE_CHECKING:
    from decent_bench.benchmark import BenchmarkProblem

MAX_ABS_METRIC_VALUE = 1e30


def compute_metrics_at_iter(
    network_views: dict[Algorithm[Network], list[NetworkMetricsView]],
    problem: "BenchmarkProblem",
    metric: Metric,
    iteration: int,
) -> pd.DataFrame:
    """
    Compute the metric at the given iteration for each algorithm and each trial.

    The function returns a DataFrame with columns (algorithm, trial, agent, iteration, value) which are of types
    (str, uint16, uint16, uint32, float32); algorithm is categorical. The elements in the
    value column are set to +/-inf if value>1e30/<-1e30.
    """
    # 1) compute metrics across algorithms and trials
    rows: list[dict[str, object]] = []

    for algorithm, network_views_by_trials in network_views.items():
        for trial_idx, network_view in enumerate(network_views_by_trials):
            data = metric.compute(network_view, problem, iteration)

            for agent_idx, value in enumerate(data):
                rows.append(
                    {
                        "algorithm": algorithm.name,
                        "trial": trial_idx,
                        "agent": agent_idx,
                        "iteration": iteration,
                        "value": value,
                    }
                )

    # 2) create dataframe, remove extreme values (or NaN), and cast columns to appropriate dtypes
    frame = pd.DataFrame.from_records(rows, columns=["algorithm", "trial", "agent", "iteration", "value"])
    frame["value"] = frame["value"].astype("float64")  # guard against pandas inferring int incorrectly

    frame.loc[
        frame["value"].isna() | (np.isfinite(frame["value"]) & (frame["value"] > MAX_ABS_METRIC_VALUE)), "value"
    ] = np.inf
    frame.loc[
        frame["value"].isna() | (np.isfinite(frame["value"]) & (frame["value"] < -MAX_ABS_METRIC_VALUE)), "value"
    ] = -np.inf

    return frame.astype(
        {"algorithm": "category", "trial": "uint16", "agent": "uint16", "iteration": "uint32", "value": "float32"}
    )
