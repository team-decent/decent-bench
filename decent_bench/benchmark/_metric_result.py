from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

from decent_bench.agents import AgentMetricsView
from decent_bench.algorithms import Algorithm
from decent_bench.metrics import Metric
from decent_bench.networks import Network

if TYPE_CHECKING:
    import pandas


def _import_pandas() -> ModuleType:
    try:
        return import_module("pandas")
    except ImportError as exc:  # pragma: no cover - exercised when pandas is unavailable
        raise ImportError("pandas is required to convert MetricResult to dataframes") from exc


@dataclass
class MetricResult:
    """
    Result of metric computation, containing metrics and computed statistics.

    This class is used to store the computed metrics and statistics from a benchmark execution.
    It is returned by the :func:`~decent_bench.benchmark.compute_metrics` function and contains
    all the information about the computed metrics, including agent-level metrics, table statistics,
    and plot data for visualization.

    * `agent_metrics`: contains the raw agent-level metrics for each algorithm, organized by algorithm where
      each algorithm maps to a sequence of trials, with each trial containing metrics for all snapshotted agents.
      For federated networks, this includes the server.
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
    plot_metrics: list[Metric] | None
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
        return sorted(
            {
                algorithm.name
                for mapping in (
                    self.agent_metrics,
                    self.table_results,
                    self.plot_results,
                )
                if mapping is not None
                for algorithm in mapping
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

    def to_dataframe(self) -> tuple["pandas.DataFrame | None", "pandas.DataFrame | None"]:
        """
        Convert the stored metric results into pandas dataframes.

        Returns:
            ``table_results``: a dataframe with columns ``algorithm``, ``metric``, ``statistic``, ``mean``, and
                ``margin_of_error``.
            ``plot_results``: a dataframe with columns ``algorithm``, ``metric``, ``x``, ``y_mean``, ``y_min``,
                and ``y_max``.

        Examples:
            Access the table dataframe and filter for a specific algorithm:

                table_df, plot_df = metric_result.to_dataframe()
                a_rows = table_df[table_df["algorithm"] == "A"]

            Access the full x and y_mean vectors for one algorithm and metric:

                table_df, plot_df = metric_result.to_dataframe()
                loss_rows = plot_df[(plot_df["algorithm"] == "A") & (plot_df["metric"] == "loss")]
                x_values = loss_rows["x"]
                y_mean_values = loss_rows["y_mean"]

        """
        pd = _import_pandas()

        table_frame = None
        if self.table_results is not None:
            table_rows: list[dict[str, object]] = []
            for algorithm, table_metric_results in self.table_results.items():
                for metric, statistics in table_metric_results.items():
                    for statistic_name, (mean, margin_of_error) in statistics.items():
                        table_rows.append({
                            "algorithm": algorithm.name,
                            "metric": metric.description,
                            "statistic": statistic_name,
                            "mean": mean,
                            "margin_of_error": margin_of_error,
                        })

            table_frame = pd.DataFrame.from_records(
                table_rows,
                columns=["algorithm", "metric", "statistic", "mean", "margin_of_error"],
            )

        plot_frame = None
        if self.plot_results is not None:
            plot_rows: list[dict[str, object]] = []
            for algorithm, plot_metric_results in self.plot_results.items():
                for metric, (x_values, y_mean_values, y_min_values, y_max_values) in plot_metric_results.items():
                    for x_value, y_mean, y_min, y_max in zip(
                        x_values,
                        y_mean_values,
                        y_min_values,
                        y_max_values,
                        strict=True,
                    ):
                        plot_rows.append({
                            "algorithm": algorithm.name,
                            "metric": metric.description,
                            "x": x_value,
                            "y_mean": y_mean,
                            "y_min": y_min,
                            "y_max": y_max,
                        })

            plot_frame = pd.DataFrame.from_records(
                plot_rows,
                columns=["algorithm", "metric", "x", "y_mean", "y_min", "y_max"],
            )

        return table_frame, plot_frame
