from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from decent_bench.agents import AgentMetricsView
from decent_bench.distributed_algorithms import Algorithm
from decent_bench.metrics import Metric


@dataclass
class MetricsResult:
    """
    TODO: Write this docstring.

    * `table_metrics`: metrics to tabulate as confidence intervals after the execution, defaults to
      :const:`~decent_bench.metrics.metric_collection.DEFAULT_TABLE_METRICS`
    * `plot_metrics`: metrics to plot after the execution, defaults to
      :const:`~decent_bench.metrics.metric_collection.DEFAULT_PLOT_METRICS`.
    * `table_results`: contains the table results for each algorithm and metric, organized by algorithm and metric.
    * `plot_results`: contains the plot results for each algorithm and metric, organized by algorithm and metric,
      where each metric maps to a tuple of sequences representing (x, y_mean, y_min, y_max) for plotting.
    """

    agent_metrics: Mapping[Algorithm, Sequence[Sequence[AgentMetricsView]]] | None
    table_metrics: list[Metric] | None
    plot_metrics: list[Metric] | list[list[Metric]] | None
    table_results: Mapping[Algorithm, Mapping[Metric, Mapping[str, tuple[float, float]]]] | None
    plot_results: (
        Mapping[Algorithm, Mapping[Metric, tuple[Sequence[float], Sequence[float], Sequence[float], Sequence[float]]]]
        | None
    )
