from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from decent_bench.agents import AgentMetricsView
from decent_bench.distributed_algorithms import Algorithm
from decent_bench.metrics import Metric
from decent_bench.networks import Network
from decent_bench.utils._metric_helpers import _flatten_plot_metrics


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

    agent_metrics: Mapping[Algorithm[Network], Sequence[Sequence[AgentMetricsView]]] | None
    table_metrics: list[Metric] | None
    plot_metrics: list[Metric] | list[list[Metric]] | None
    table_results: Mapping[Algorithm[Network], Mapping[Metric, Mapping[str, tuple[float, float]]]] | None
    plot_results: (
        Mapping[
            Algorithm[Network],
            Mapping[Metric, tuple[Sequence[float], Sequence[float], Sequence[float], Sequence[float]]],
        ]
        | None
    )

    @property
    def available_algorithms(self) -> list[str]:
        """Return ``name`` of available algorithms, which can be used for filtering in :func:`~decent_bench.benchmark.display_metrics`."""  # noqa: E501
        return sorted({
            algorithm.name
            for mapping in (self.agent_metrics, self.table_results, self.plot_results)
            if mapping is not None
            for algorithm in mapping
        })

    @property
    def available_table_metrics(self) -> list[str]:
        """Return ``table_description`` of available metrics, which can be used for filtering in :func:`~decent_bench.benchmark.display_metrics`."""  # noqa: E501
        return sorted({metric.table_description for metric in (self.table_metrics or [])})

    @property
    def available_plot_metrics(self) -> list[str]:
        """Return ``plot_descriptions`` of available metrics, which can be used for filtering in :func:`~decent_bench.benchmark.display_metrics`."""  # noqa: E501
        return sorted({metric.plot_description for metric in (_flatten_plot_metrics(self.plot_metrics or []))})
