import warnings
from typing import Literal

import numpy as np
import tabulate
from scipy import stats

from decent_bench.library.core.agent import AgentMetricsView
from decent_bench.library.core.benchmark_problem.benchmark_problems import BenchmarkProblem
from decent_bench.library.core.dst_algorithms import DstAlgorithm
from decent_bench.library.core.metrics.table_metrics.table_metrics_constructs import TableMetric
from decent_bench.library.core.network import Network


def generate_table(
    resulting_nw_states_per_alg: dict[DstAlgorithm, list[Network]],
    problem: BenchmarkProblem,
    metrics: list[TableMetric],
    confidence_level: float,
    table_fmt: Literal["grid", "latex"],
) -> str:
    """
    Generate table with confidence intervals, one row per metric and statistic, and one column per algorithm.

    Args:
        resulting_nw_states_per_alg: resulting network states from the trial executions, ordered by algorithm
        problem: benchmark problem whose properties, e.g.
            :attr:`~decent_bench.library.core.benchmark_problem.benchmark_problems.BenchmarkProblem.optimal_x`,
            are used for metric calculations
        metrics: metrics to calculate
        confidence_level: confidence level of the confidence intervals
        table_fmt: table format, grid is suitable for the terminal while latex can be copy-pasted into a latex document

    Returns:
        formatted table with confidence intervals, one row per metric and statistic, and one column per algorithm

    """
    with warnings.catch_warnings(action="ignore"):
        headers = ["Metric (statistic)"] + [alg.name for alg in resulting_nw_states_per_alg]
        rows: list[list[str]] = []
        for metric in metrics:
            for statistic in metric.statistics:
                row = [f"{metric.name} ({statistic.name})"]
                for resulting_nw_states in resulting_nw_states_per_alg.values():
                    aggregated_data_per_trial: list[float] = []
                    for nw in resulting_nw_states:
                        agent_metrics_views = [AgentMetricsView.from_agent(a) for a in nw.get_all_agents()]
                        trial_data = metric.get_data_from_trial(agent_metrics_views, problem)
                        aggregated_trial_data = statistic.agg_func(trial_data)
                        aggregated_data_per_trial.append(aggregated_trial_data)
                    mean, margin_of_error = _get_mean_and_margin_of_error(aggregated_data_per_trial, confidence_level)
                    formatted_confidence_interval = _format_confidence_interval(mean, margin_of_error)
                    row.append(formatted_confidence_interval)
                rows.append(row)
        return tabulate.tabulate(rows, headers, tablefmt=table_fmt)


def _get_mean_and_margin_of_error(data: list[float], confidence_level: float) -> tuple[float, float]:
    mean = np.mean(data)
    sem = stats.sem(data) if len(set(data)) > 1 else None
    raw_interval = (
        stats.t.interval(confidence=confidence_level, df=len(data) - 1, loc=mean, scale=sem) if sem else (mean, mean)
    )
    res = (float(mean), float(mean - raw_interval[0]))
    if any(np.isinf(res)):
        return np.nan, np.nan
    return res


def _format_confidence_interval(mean: float, margin_of_error: float) -> str:
    formatted_confidence_interval = f"{mean:.3e} \u00b1 {margin_of_error:.3e}"
    if any(np.isnan([mean, margin_of_error])):
        formatted_confidence_interval += " (diverged?)"
    return formatted_confidence_interval
