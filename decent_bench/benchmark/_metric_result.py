from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from decent_bench.agents import AgentMetricsView
from decent_bench.distributed_algorithms import Algorithm
from decent_bench.metrics import Metric


@dataclass
class MetricResult:
    """
    Result of metric computation, containing metrics and computed statistics.

    This class is used to store the computed metrics and statistics from a benchmark execution.
    It is returned by the :func:`~decent_bench.benchmark.compute_metrics` function and contains
    all the information about the computed metrics, including agent-level metrics, table statistics,
    and plot data for visualization.

    * `agent_metrics`: contains the raw agent-level metrics for each algorithm, organized by algorithm where
      each algorithm maps to a sequence of trials, with each trial containing metrics for all agents.
    * `table_metrics`: contains the list of metrics that were tabulated as confidence intervals.
    * `plot_metrics`: contains the list of metrics that were plotted.
    * `table_results`: contains the computed table statistics for each algorithm and metric, organized by
      algorithm and metric, where each metric maps to statistics with their confidence intervals.
    * `plot_results`: contains the plot data for each algorithm and metric, organized by algorithm and metric,
      where each metric maps to a tuple of sequences representing (x, y_mean, y_min, y_max) for plotting.

    These results can be used for analysis, visualization, and comparison of the algorithms' performance on the
    benchmark problem.
    """

    agent_metrics: Mapping[Algorithm, Sequence[Sequence[AgentMetricsView]]] | None
    table_metrics: list[Metric] | None
    plot_metrics: list[Metric] | list[list[Metric]] | None
    table_results: Mapping[Algorithm, Mapping[Metric, Mapping[str, tuple[float, float]]]] | None
    plot_results: (
        Mapping[Algorithm, Mapping[Metric, tuple[Sequence[float], Sequence[float], Sequence[float], Sequence[float]]]]
        | None
    )
