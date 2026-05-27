import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

import decent_bench.metrics.metric_utils as utils
from decent_bench.algorithms import Algorithm
from decent_bench.metrics._metric import Metric
from decent_bench.metrics._metrics_view import NetworkMetricsView
from decent_bench.networks import Network

if TYPE_CHECKING:
    from decent_bench.benchmark import BenchmarkProblem

MAX_ABS_METRIC_VALUE = 1e30


def compute_table_metrics(
    network_views: dict[Algorithm[Network], list[NetworkMetricsView]],
    problem: "BenchmarkProblem",
    metrics: list[Metric],
) -> Mapping[Metric, pd.DataFrame]:
    """Compute raw table metric frames with MultiIndex (algorithm, trial, agents)."""
    if not metrics:
        return {}

    raw_results: dict[Metric, pd.DataFrame] = {}
    total_tasks = len(metrics) * len(network_views)

    with utils.MetricProgressBar() as progress:
        table_task = progress.add_task("Computing table metrics", total=total_tasks, status="")

        for metric in metrics:
            progress.update(table_task, status=f"Task: {metric.description}")
            rows: list[dict[str, object]] = []

            for algorithm, alg_network_views in network_views.items():
                for trial_idx, network_view in enumerate(alg_network_views):
                    with warnings.catch_warnings(action="ignore"):
                        data = metric.get_table_data(network_view, problem)

                    if len(data) == 1:
                        rows.append(
                            {
                                "algorithm": algorithm.name,
                                "trial": trial_idx,
                                "agent": np.nan,
                                "value": float(data[0]),
                            }
                        )
                    for agent_idx, value in enumerate(data):
                        rows.append(
                            {
                                "algorithm": algorithm.name,
                                "trial": trial_idx,
                                "agent": agent_idx,
                                "value": float(value),
                            }
                        )

                progress.advance(table_task)

            frame = pd.DataFrame.from_records(rows, columns=["algorithm", "trial", "agent", "value"])
            frame["value"] = frame["value"].mask(frame["value"] > MAX_ABS_METRIC_VALUE, np.inf)
            frame["value"] = frame["value"].mask(frame["value"] < -MAX_ABS_METRIC_VALUE, -np.inf)
            frame["value"] = frame["value"].astype("float32")
            frame = frame.set_index(["algorithm", "trial", "agent"]).sort_index()
            raw_results[metric] = frame

        progress.update(table_task, status="Table computation complete")

    return raw_results
