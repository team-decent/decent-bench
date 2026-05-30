import pytest

from decent_bench.benchmark import MetricResult
from decent_bench.metrics._metric import Metric


class _AlgorithmStub:
    def __init__(self, name: str) -> None:
        self.name = name


class _MetricStub(Metric):
    def __init__(self, description: str) -> None:
        super().__init__(fmt=".2e", y_log=False)
        self._description = description

    @property
    def description(self) -> str:
        return self._description

    def compute(self, agents, problem, iteration):  # noqa: D102
        return [0.0]


def test_metric_result_update_tables_computes_dataframe() -> None:
    pd = pytest.importorskip("pandas")

    metric_one = _MetricStub("one")

    metrics_result = MetricResult(
        network_views={_AlgorithmStub("A"): [], _AlgorithmStub("B"): []},
        raw_table_results={
            metric_one: pd.DataFrame(
                {
                    "algorithm": ["A", "A", "A", "A"],
                    "trial": [0, 0, 1, 1],
                    "agent": [0, 1, 0, 1],
                    "value": [1.0, 2.0, 3.0, 5.0],
                }
            ),
        },
        raw_plot_results=None,
        table_results=None,
        plot_results=None,
    )

    table_df = metrics_result.update_table_results(["mean", "std"])

    assert table_df is not None
    assert list(table_df.columns) == ["metric", "algorithm", "statistic", "mean", "std"]
    assert table_df["mean"].dtype == "float32"
    assert table_df["std"].dtype == "float32"
    mean_row = table_df[
        (table_df["metric"] == "one") & (table_df["statistic"] == "mean") & (table_df["algorithm"] == "A")
    ].iloc[0]
    std_row = table_df[
        (table_df["metric"] == "one") & (table_df["statistic"] == "std") & (table_df["algorithm"] == "A")
    ].iloc[0]
    assert float(mean_row["mean"]) == pytest.approx(2.75)
    assert float(std_row["mean"]) == pytest.approx(0.75)


def test_metric_result_algorithms_uses_raw_frames_when_network_views_missing() -> None:
    pd = pytest.importorskip("pandas")

    metric = _MetricStub("metric")
    result = MetricResult(
        network_views=None,
        raw_table_results={
            metric: pd.DataFrame(
                {
                    "algorithm": ["A", "B"],
                    "trial": [0, 0],
                    "agent": [0, 0],
                    "value": [1.0, 2.0],
                }
            )
        },
        raw_plot_results=None,
        table_results=None,
        plot_results=None,
    )

    assert result.algorithms == ["A", "B"]
