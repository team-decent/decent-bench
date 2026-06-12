from collections.abc import Mapping
from typing import TYPE_CHECKING

import pandas as pd

from decent_bench.algorithms import Algorithm
from decent_bench.benchmark._compute.compute_metrics_at_iter import compute_metrics_at_iter
from decent_bench.metrics import Metric, utils
from decent_bench.metrics._metrics_view import NetworkMetricsView
from decent_bench.networks import Network

if TYPE_CHECKING:
    from decent_bench.benchmark import BenchmarkProblem


def compute_plot_metrics(
    network_views: dict[Algorithm[Network], list[NetworkMetricsView]],
    problem: "BenchmarkProblem",
    metrics: list[Metric],
    iterations: list[int],
) -> Mapping[Metric, pd.DataFrame]:
    """
    Compute metrics at each iteration and return one DataFrame per metric.

    The DataFrame has columns (algorithm, trial, agent, iteration, value).
    """
    frames_by_metric: dict[Metric, pd.DataFrame] = {}
    if not iterations:
        return frames_by_metric
    total_tasks = len(metrics) * len(iterations)

    with utils.MetricProgressBar() as progress:
        plot_task = progress.add_task("Computing plot metrics", total=total_tasks, status="")

        for metric in metrics:
            progress.update(plot_task, status=f"Task: {metric.description}")

            frames_by_iterations = []
            for iteration in iterations:
                frames_by_iterations.append(compute_metrics_at_iter(network_views, problem, metric, iteration))
                progress.advance(plot_task)

            frames_by_metric[metric] = pd.concat(frames_by_iterations, ignore_index=True)

        progress.update(plot_task, status="Plot computation complete")

    return frames_by_metric


def aggregate_plot_metrics(
    plot_results: Mapping[Metric, pd.DataFrame],
) -> pd.DataFrame | None:
    """
    Aggregate plot metrics by mean across agents and by mean, min, and max across trials.

    The DataFrame that is returned has columns (metric, algorithm, iteration, mean, min, max), with mean, min, max being
    float32.
    """
    frames_by_metric: list[pd.DataFrame] = []

    for metric, frame in plot_results.items():
        # 1) take mean of values across agents
        new_frame = frame.groupby(["algorithm", "trial", "iteration"])["value"].mean().reset_index()

        # 2) compute mean, min, max across trials
        new_frame = (
            new_frame.groupby(["algorithm", "iteration"])["value"].agg(mean="mean", min="min", max="max").reset_index()
        )

        # 3) add metric column
        new_frame = new_frame.assign(metric=metric.description)
        new_frame = new_frame[["metric", "algorithm", "iteration", "mean", "min", "max"]]  # reorder columns
        new_frame[["mean", "min", "max"]] = new_frame[["mean", "min", "max"]].astype("float32")

        frames_by_metric.append(new_frame)

    # 4) return concatenated frames
    if not frames_by_metric:
        return None
    return pd.concat(frames_by_metric, ignore_index=True)
