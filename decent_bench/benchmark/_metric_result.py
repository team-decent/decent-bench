from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from decent_bench.agents import AgentMetricsView
from decent_bench.distributed_algorithms import Algorithm
from decent_bench.metrics import Metric
from decent_bench.networks import Network


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
        names: list[str] = []
        seen: set[str] = set()

        def _collect(mapping: Mapping[Algorithm[Network], object] | None) -> None:
            if mapping is None:
                return
            for algorithm in mapping:
                if algorithm.name not in seen:
                    seen.add(algorithm.name)
                    names.append(algorithm.name)

        _collect(cast("Mapping[Algorithm[Network], object] | None", self.agent_metrics))
        _collect(cast("Mapping[Algorithm[Network], object] | None", self.table_results))
        _collect(cast("Mapping[Algorithm[Network], object] | None", self.plot_results))

        return names

    @property
    def available_table_metrics(self) -> list[str]:
        """Return ``table_description`` of available metrics, which can be used for filtering in :func:`~decent_bench.benchmark.display_metrics`."""  # noqa: E501
        if self.table_metrics is None:
            return []

        descriptions: list[str] = []
        seen: set[str] = set()
        for metric in self.table_metrics:
            if metric.table_description not in seen:
                seen.add(metric.table_description)
                descriptions.append(metric.table_description)

        return descriptions

    @property
    def available_plot_metrics(self) -> list[str]:
        """Return ``plot_descriptions`` of available metrics, which can be used for filtering in :func:`~decent_bench.benchmark.display_metrics`."""  # noqa: E501
        if self.plot_metrics is None:
            return []

        descriptions: list[str] = []
        seen: set[str] = set()
        for metric in self._flatten_plot_metrics(self.plot_metrics):
            if metric.plot_description not in seen:
                seen.add(metric.plot_description)
                descriptions.append(metric.plot_description)

        return descriptions

    @staticmethod
    def _flatten_plot_metrics(plot_metrics: list[Metric] | list[list[Metric]]) -> list[Metric]:
        if any(isinstance(metric, list) for metric in plot_metrics):
            return [metric for group in cast("list[list[Metric]]", plot_metrics) for metric in group]

        return cast("list[Metric]", plot_metrics)
