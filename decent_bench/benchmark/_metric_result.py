from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from decent_bench.algorithms import Algorithm
from decent_bench.metrics import Metric
from decent_bench.metrics._metrics_view import NetworkMetricsView
from decent_bench.metrics.tables.compute_tables import compute_tables
from decent_bench.networks import Network


@dataclass
class MetricResult:
    """
    Result of metric computation, containing metrics and computed statistics.

    This class is used to store the computed metrics and statistics from a benchmark execution.
    It is returned by the :func:`~decent_bench.benchmark.compute_metrics` function and contains
    all the information about the computed metrics, including agent-level metrics, table statistics,
    and plot data for visualization.

    * `network_views`: contains the raw network-level metrics for each algorithm, organized by algorithm where
        each algorithm maps to a sequence of trials, with each trial containing a
        :class:`~decent_bench.metrics.NetworkMetricsView`.
    * `table_metrics`: contains the list of metrics that were tabulated as mean+/-std intervals.
    * `plot_metrics`: contains the list of metrics that were plotted.
    * `table_results`: contains the computed table statistics for each algorithm and metric, organized by
        algorithm and metric, where each metric maps to statistics with mean+/-std.
    * `plot_results`: contains the plot data for each algorithm and metric, organized by algorithm and metric,
        where each metric maps to a tuple of sequences representing (x, y_mean, y_min, y_max) for plotting.

    These results can be used for analysis, visualization, and comparison of the algorithms' performance on the
    benchmark problem.
    """

    network_views: Mapping[Algorithm[Network], Sequence[NetworkMetricsView]] | None
    table_metrics: list[Metric] | None
    plot_metrics: list[Metric] | None
    raw_table_results: Mapping[Metric, pd.DataFrame] | None
    raw_plot_results: Mapping[Metric, pd.DataFrame] | None
    table_results: pd.DataFrame | None = None
    plot_results: pd.DataFrame | None = None

    def update_table_results(
        self,
        statistics_across_agents: list[str] | None,
    ) -> pd.DataFrame:
        """Recompute aggregated table statistics from stored raw table results."""
        self.table_results = compute_tables(self, statistics_across_agents)
        return self.table_results

    @property
    def available_algorithms(self) -> list[str]:
        """Return ``name`` of available algorithms, which can be used for filtering in :func:`~decent_bench.benchmark.display_metrics`."""  # noqa: E501
        return sorted(
            {
                algorithm.name
                for mapping in (
                    self.network_views,
                )
                if mapping is not None
                for algorithm in mapping
            }
            |
            (
                {
                    str(name)
                    for name in self.table_results.index.get_level_values("algorithm").unique()
                }
                if isinstance(self.table_results, pd.DataFrame) and isinstance(self.table_results.index, pd.MultiIndex)
                else set()
            )
            |
            (
                {
                    str(name)
                    for name in self.plot_results.index.get_level_values("algorithm").unique()
                }
                if isinstance(self.plot_results, pd.DataFrame) and isinstance(self.plot_results.index, pd.MultiIndex)
                else set()
            )
            |
            {
                str(name)
                for frame in (self.raw_table_results or {}).values()
                if isinstance(frame.index, pd.MultiIndex) and "algorithm" in frame.index.names
                for name in frame.index.get_level_values("algorithm").unique()
            }
            |
            {
                str(name)
                for frame in (self.raw_plot_results or {}).values()
                if isinstance(frame.index, pd.MultiIndex) and "algorithm" in frame.index.names
                for name in frame.index.get_level_values("algorithm").unique()
            }
        )

    @property
    def available_table_metrics(self) -> list[str]:
        """Return ``description`` of available table metrics, which can be used for filtering in :func:`~decent_bench.benchmark.display_metrics`."""  # noqa: E501
        return sorted({metric.description for metric in (self.table_metrics or [])})

    @property
    def available_plot_metrics(self) -> list[str]:
        """Return ``description`` of available plot metrics, which can be used for filtering in :func:`~decent_bench.benchmark.display_metrics`."""  # noqa: E501
        return sorted({metric.description for metric in (self.plot_metrics or [])})
