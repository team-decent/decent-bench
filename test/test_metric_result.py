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

    algorithm_a = _AlgorithmStub("A")
    algorithm_b = _AlgorithmStub("B")
    metric_one = _MetricStub("one")

    metrics_result = MetricResult(
        network_views={algorithm_a: [], algorithm_b: []},
        table_metrics=[metric_one],
        plot_metrics=None,
        raw_table_results={
            metric_one: pd.DataFrame(
                {
                    "value": [1.0, 2.0, 3.0, 5.0],
                },
                index=pd.MultiIndex.from_tuples(
                    [("A", 0, 0), ("A", 0, 1), ("A", 1, 0), ("A", 1, 1)],
                    names=["algorithm", "trial", "agent"],
                ),
            ),
        },
        table_results=None,
        plot_results=None,
    )

    table_df = metrics_result.update_tables(["mean", "std"])

    assert table_df is not None
    assert list(table_df.columns) == ["mean", "std"]
    assert table_df.index.names == ["metric", "statistic", "algorithm"]
    assert table_df["mean"].dtype == "float32"
    assert table_df["std"].dtype == "float32"
    assert float(table_df.loc[("one", "mean", "A"), "mean"]) == pytest.approx(2.75)
    assert float(table_df.loc[("one", "std", "A"), "mean"]) == pytest.approx(0.75)


def test_metric_result_available_algorithms_uses_dataframe_indices() -> None:
    pd = pytest.importorskip("pandas")

    result = MetricResult(
        network_views=None,
        table_metrics=[],
        plot_metrics=[],
        raw_table_results=None,
        raw_plot_results=None,
        table_results=pd.DataFrame(
            {"mean": [1.0], "std": [0.1]},
            index=pd.MultiIndex.from_tuples(
                [("metric-a", "mean", "A")], names=["metric", "statistic", "algorithm"]
            ),
        ),
        plot_results=pd.DataFrame(
            {"y_mean": [1.0], "y_min": [0.9], "y_max": [1.1]},
            index=pd.MultiIndex.from_tuples(
                [("metric-b", "B", 1)], names=["metric", "algorithm", "iterations"]
            ),
        ),
    )

    assert sorted(result.available_algorithms) == ["A", "B"]
