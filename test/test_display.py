import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import importlib
from copy import deepcopy
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score
from types import SimpleNamespace
from uuid import uuid4

from decent_bench.agents import Agent, AgentHistory
from decent_bench.algorithms.federated import FedAvg
from decent_bench.benchmark import BenchmarkProblem, BenchmarkResult, compute_metrics, display_metrics
from decent_bench.benchmark._metric_result import MetricResult
from decent_bench.costs import LinearRegressionCost, LogisticRegressionCost, QuadraticCost
from decent_bench.metrics._metric import Metric
from decent_bench.metrics._metrics_view import AgentMetricsView, NetworkMetricsView, NetworkType
from decent_bench.metrics import metric_library as ml
from decent_bench.benchmark._compute.compute_metrics_at_iter import MAX_ABS_METRIC_VALUE
from decent_bench.benchmark._compute.compute_plots import aggregate_plot_metrics, compute_plot_metrics
from decent_bench.benchmark._display.display_plots import (
    _add_legend_and_save,
    _create_separate_legend_figure,
    _get_separate_legend_path,
    _select_legend_mode,
)
from decent_bench.benchmark._display.display_tables import display_tables
from decent_bench.metrics.metric_library import Accuracy, BalancedAccuracy, MSE, Precision, Recall, Regret, XError
from decent_bench.networks import FedNetwork

display_plots_module = importlib.import_module("decent_bench.benchmark._display.display_plots")


# -----------------------------------------------------------------------------
# Test Helpers
# -----------------------------------------------------------------------------


class _AlgorithmStub:
    def __init__(self, name: str) -> None:
        self.name = name


class _MetricStub(Metric):
    def __init__(
        self,
        description: str,
    ) -> None:
        super().__init__(fmt=".2e", y_log=False)
        self._description = description

    @property
    def description(self) -> str:
        return self._description

    def compute(self, agents, problem, iteration):  # noqa: D102
        return [0.0]


class _PredictCostStub:
    def __init__(self, predictions: list[int]) -> None:
        self.predictions = predictions

    def predict(self, x: object, data: list[object]) -> list[int]:  # noqa: ARG002
        return self.predictions


class _ProblemStub:
    def __init__(self, labels: list[int]) -> None:
        self.test_data = [(np.array([index]), label) for index, label in enumerate(labels)]


def _agent_metrics_view(x_value: float) -> AgentMetricsView:
    history = AgentHistory()
    history[0] = np.array([x_value])
    return AgentMetricsView(
        id=uuid4(),
        cost=object(),
        x_history=history,
        n_x_updates=0,
        n_function_calls=0.0,
        n_gradient_calls=0.0,
        n_hessian_calls=0.0,
        n_proximal_calls=0.0,
        n_sent_messages=0,
        n_received_messages=0,
        n_sent_messages_dropped=0,
        n_times_selected=0,
    )


def _network_metrics_view(
    agents: list[AgentMetricsView],
    *,
    network_type: NetworkType = NetworkType.P2P,
    server: AgentMetricsView | None = None,
) -> NetworkMetricsView:
    graph = nx.Graph()
    graph.add_nodes_from(agents)
    return NetworkMetricsView(graph=graph, network_type=network_type, _server=server)


def _run_display_with_capture(
    monkeypatch,
    metrics_result: MetricResult,
    **display_kwargs,
) -> dict[str, MetricResult]:
    captured: dict[str, MetricResult] = {}
    display_metrics_module = importlib.import_module("decent_bench.benchmark._display._display_metrics")

    def _snapshot_metric_result(source: MetricResult) -> MetricResult:
        """Capture display-time state so later restoration does not affect assertions."""
        return deepcopy(source)

    def _capture_tables(
        passed_metrics_result: MetricResult,
        table_fmt: str = "grid",
        scale_compute: float = 1.0,
        table_path=None,
    ) -> None:
        captured["table"] = _snapshot_metric_result(passed_metrics_result)

    def _capture_plots(
        passed_metrics_result: MetricResult,
        *,
        computational_cost,
        scale_x_axis: float = 1e-4,
        compare_iterations_and_computational_cost: bool = False,
        individual_plots: bool = False,
        plot_grid: bool = True,
        plot_format: str = "png",
        plot_path=None,
        show_plots=True,
    ) -> None:
        captured["plot"] = _snapshot_metric_result(passed_metrics_result)

    monkeypatch.setattr(display_metrics_module, "display_tables", _capture_tables)
    monkeypatch.setattr(display_metrics_module, "display_plots", _capture_plots)

    try:
        display_metrics(metrics_result=metrics_result, **display_kwargs)
    finally:
        plt.close("all")
    return captured


def _build_minimal_benchmark_result() -> BenchmarkResult:
    history = AgentHistory()
    history[0] = np.array([0.0])

    fake_agent = SimpleNamespace(
        cost=object(),
        _x_history=history,
        _n_x_updates=0,
        _n_function_calls=0.0,
        _n_gradient_calls=0.0,
        _n_hessian_calls=0.0,
        _n_proximal_calls=0.0,
        _n_sent_messages=0,
        _n_received_messages=0,
        _n_sent_messages_dropped=0,
        _n_times_selected=0,
    )
    fake_network = SimpleNamespace(agents=lambda: [fake_agent])
    algorithm = _AlgorithmStub("A")
    return BenchmarkResult(
        problem=BenchmarkProblem(network=fake_network),
        states={algorithm: [fake_network]},
    )


def _build_federated_benchmark_result(iterations: int = 2) -> tuple[BenchmarkResult, FedAvg]:
    clients = [
        Agent(QuadraticCost(np.eye(1), np.array([0.0]))),
        Agent(QuadraticCost(np.eye(1), np.array([0.0]))),
    ]
    network = FedNetwork(clients=clients)
    algorithm = FedAvg(iterations=iterations, step_size=0.1)
    algorithm.run(network)
    return BenchmarkResult(problem=BenchmarkProblem(network=network), states={algorithm: [network]}), algorithm


def _build_display_metric_result(
    algorithms: list[_AlgorithmStub],
    table_metrics: list[_MetricStub],
    plot_metrics: list[_MetricStub],
    *,
    agent_x_values: list[float] | None = None,
    table_results: pd.DataFrame | None = None,
    raw_table_results: dict[_MetricStub, pd.DataFrame] | None = None,
    raw_plot_results: dict[_MetricStub, pd.DataFrame] | None = None,
    plot_results: pd.DataFrame | None = None,
) -> MetricResult:
    if agent_x_values is None:
        agent_x_values = [1.0] * len(algorithms)

    default_table_rows: list[dict[str, object]] = []
    raw_table_records: dict[_MetricStub, list[tuple[str, int, int, float]]] = {}
    default_plot_rows: list[dict[str, object]] = []

    for alg_idx, alg in enumerate(algorithms):
        for metric_idx, metric in enumerate(table_metrics):
            value = float(alg_idx + metric_idx + 1)
            default_table_rows.append(
                {
                    "metric": metric.description,
                    "statistic": "avg",
                    "algorithm": alg.name,
                    "mean": value,
                    "std": 0.0,
                }
            )
            raw_table_records.setdefault(metric, []).append((alg.name, 0, 0, value))
        for metric_idx, metric in enumerate(plot_metrics):
            value = float(alg_idx + metric_idx + 1)
            default_plot_rows.append(
                {
                    "metric": metric.description,
                    "algorithm": alg.name,
                    "iteration": 0.0,
                    "mean": value,
                    "min": value,
                    "max": value,
                }
            )

    default_raw_table_results = {
        metric: pd.DataFrame(
            {
                "algorithm": [record[0] for record in records],
                "trial": [record[1] for record in records],
                "agent": [record[2] for record in records],
                "value": [record[3] for record in records],
            }
        )
        for metric, records in raw_table_records.items()
    }

    default_raw_plot_results = {
        metric: pd.DataFrame(
            {
                "algorithm": [alg.name for alg in algorithms],
                "trial": [0 for _ in algorithms],
                "agent": [0 for _ in algorithms],
                "iteration": [0 for _ in algorithms],
                "value": [float(alg_idx + metric_idx + 1) for alg_idx, _ in enumerate(algorithms)],
            }
        )
        for metric_idx, metric in enumerate(plot_metrics)
    }

    return MetricResult(
        network_views={
            alg: [_network_metrics_view([_agent_metrics_view(agent_x_values[idx])])]
            for idx, alg in enumerate(algorithms)
        },
        raw_table_results=raw_table_results or default_raw_table_results,
        raw_plot_results=raw_plot_results or default_raw_plot_results,
        table_results=table_results
        if table_results is not None
        else pd.DataFrame.from_records(default_table_rows, columns=["metric", "statistic", "algorithm", "mean", "std"]),
        plot_results=plot_results
        if plot_results is not None
        else pd.DataFrame.from_records(
            default_plot_rows, columns=["metric", "algorithm", "iteration", "mean", "min", "max"]
        ),
    )


# -----------------------------------------------------------------------------
# display_metrics Filtering Tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("algorithm_filter", "expected_algorithms"),
    [
        ("object", ["B"]),
        ("name", ["A"]),
    ],
)
def test_display_metrics_filters_algorithms(monkeypatch, algorithm_filter: str, expected_algorithms: list[str]) -> None:  # noqa: D103
    alg_a = _AlgorithmStub("A")
    alg_b = _AlgorithmStub("B")
    metric = _MetricStub("one")

    metrics_result = _build_display_metric_result([alg_a, alg_b], [metric], [metric])
    algorithms = [alg_b] if algorithm_filter == "object" else ["A"]
    captured = _run_display_with_capture(monkeypatch, metrics_result, algorithms=algorithms)

    assert "table" in captured
    assert "plot" in captured
    assert [alg.name for alg in captured["table"].network_views] == expected_algorithms
    assert captured["table"].raw_table_results is not None
    raw_metric_df = next(iter(captured["table"].raw_table_results.values()))
    assert raw_metric_df["algorithm"].unique().tolist() == expected_algorithms
    assert metrics_result.table_results is not None
    assert metrics_result.table_results["algorithm"].unique().tolist() == ["A", "B"]


def test_display_metrics_keeps_nan_table_metrics(monkeypatch) -> None:  # noqa: D103
    alg_a = _AlgorithmStub("A")
    alg_b = _AlgorithmStub("B")
    valid_metric = _MetricStub("valid")
    nan_metric = _MetricStub("nan")

    metrics_result = _build_display_metric_result(
        [alg_a, alg_b],
        [valid_metric, nan_metric],
        [valid_metric, nan_metric],
        agent_x_values=[1.0, 2.0],
        table_results=pd.DataFrame.from_records(
            [
                {"metric": valid_metric.description, "statistic": "avg", "algorithm": "A", "mean": 1.0, "std": 0.1},
                {"metric": nan_metric.description, "statistic": "avg", "algorithm": "A", "mean": np.nan, "std": np.nan},
                {"metric": valid_metric.description, "statistic": "avg", "algorithm": "B", "mean": 2.0, "std": 0.1},
                {"metric": nan_metric.description, "statistic": "avg", "algorithm": "B", "mean": np.nan, "std": np.nan},
            ]
        ),
        plot_results=pd.DataFrame.from_records(
            [
                {"metric": valid_metric.description, "algorithm": "A", "iteration": 0.0, "mean": 1.0, "min": 1.0, "max": 1.0},
                {"metric": valid_metric.description, "algorithm": "B", "iteration": 0.0, "mean": 2.0, "min": 2.0, "max": 2.0},
            ]
        ),
    )

    captured = _run_display_with_capture(monkeypatch, metrics_result)

    assert "table" in captured
    assert "plot" in captured
    assert captured["table"].table_metrics == ["nan", "valid"]
    assert captured["table"].plot_metrics == ["nan", "valid"]
    assert captured["table"].table_results["metric"].unique().tolist() == ["valid", "nan"]


def test_display_metrics_filters_metrics_by_name(monkeypatch) -> None:  # noqa: D103
    alg_a = _AlgorithmStub("A")
    metric_1 = _MetricStub("one")
    metric_2 = _MetricStub("two")

    metrics_result = _build_display_metric_result([alg_a], [metric_1, metric_2], [metric_1, metric_2])

    captured = _run_display_with_capture(
        monkeypatch,
        metrics_result,
        table_metrics=["two"],
        plot_metrics=["one"],
    )

    assert "table" in captured
    assert captured["table"].table_metrics == ["two"]
    assert captured["table"].plot_metrics == ["one"]


def test_display_metrics_filters_metrics_with_mixed_objects_and_names(monkeypatch) -> None:  # noqa: D103
    alg_a = _AlgorithmStub("A")
    metric_1 = _MetricStub("one")
    metric_2 = _MetricStub("two")

    metrics_result = _build_display_metric_result([alg_a], [metric_1, metric_2], [metric_1, metric_2])

    captured = _run_display_with_capture(
        monkeypatch,
        metrics_result,
        table_metrics=[metric_1, "two"],
        plot_metrics=[metric_1, "two"],
    )

    assert "table" in captured
    assert captured["table"].table_metrics == ["one", "two"]
    assert captured["table"].plot_metrics == ["one", "two"]


def test_display_metrics_filters_algorithms_with_mixed_objects_and_names(monkeypatch) -> None:  # noqa: D103
    alg_a = _AlgorithmStub("A")
    alg_b = _AlgorithmStub("B")
    alg_c = _AlgorithmStub("C")
    metric = _MetricStub("one")

    metrics_result = _build_display_metric_result([alg_a, alg_b, alg_c], [metric], [metric])

    captured = _run_display_with_capture(monkeypatch, metrics_result, algorithms=[alg_a, "C"])

    assert "table" in captured
    assert [alg.name for alg in captured["table"].network_views] == ["A", "C"]
    assert captured["table"].table_results["algorithm"].unique().tolist() == ["A", "C"]
    assert captured["table"].plot_results["algorithm"].unique().tolist() == ["A", "C"]
    assert metrics_result.table_results["algorithm"].unique().tolist() == ["A", "B", "C"]


def test_display_metrics_returns_early_when_all_algorithms_filtered_out(monkeypatch) -> None:  # noqa: D103
    alg_a = _AlgorithmStub("A")
    metric = _MetricStub("one")

    metrics_result = MetricResult(
        network_views={alg_a: [_network_metrics_view([_agent_metrics_view(1.0)])]},
        raw_table_results={
            metric: pd.DataFrame({"algorithm": ["A"], "trial": [0], "agent": [0], "value": [1.0]})
        },
        raw_plot_results={
            metric: pd.DataFrame(
                {"algorithm": ["A"], "trial": [0], "agent": [0], "iteration": [0], "value": [1.0]}
            )
        },
        table_results=pd.DataFrame.from_records(
            [{"metric": metric.description, "statistic": "avg", "algorithm": "A", "mean": 1.0, "std": 0.0}]
        ),
        plot_results=pd.DataFrame.from_records(
            [{"metric": metric.description, "algorithm": "A", "iteration": 0.0, "mean": 1.0, "min": 1.0, "max": 1.0}]
        ),
    )

    captured = _run_display_with_capture(monkeypatch, metrics_result, algorithms=["NonExistent"])
    assert captured == {}


def test_display_metrics_returns_early_when_all_table_and_plot_metrics_filtered_out(monkeypatch) -> None:  # noqa: D103
    alg_a = _AlgorithmStub("A")
    metric_1 = _MetricStub("one")
    metric_2 = _MetricStub("two")

    metrics_result = MetricResult(
        network_views={alg_a: [_network_metrics_view([_agent_metrics_view(1.0)])]},
        raw_table_results={
            metric_1: pd.DataFrame({"algorithm": ["A"], "trial": [0], "agent": [0], "value": [1.0]}),
            metric_2: pd.DataFrame({"algorithm": ["A"], "trial": [0], "agent": [0], "value": [2.0]}),
        },
        raw_plot_results={
            metric_1: pd.DataFrame(
                {"algorithm": ["A"], "trial": [0], "agent": [0], "iteration": [0], "value": [1.0]}
            ),
            metric_2: pd.DataFrame(
                {"algorithm": ["A"], "trial": [0], "agent": [0], "iteration": [0], "value": [2.0]}
            ),
        },
        table_results=pd.DataFrame.from_records(
            [
                {"metric": metric_1.description, "statistic": "avg", "algorithm": "A", "mean": 1.0, "std": 0.0},
                {"metric": metric_2.description, "statistic": "avg", "algorithm": "A", "mean": 2.0, "std": 0.0},
            ]
        ),
        plot_results=pd.DataFrame.from_records(
            [
                {"metric": metric_1.description, "algorithm": "A", "iteration": 0.0, "mean": 1.0, "min": 1.0, "max": 1.0},
                {"metric": metric_2.description, "algorithm": "A", "iteration": 0.0, "mean": 2.0, "min": 2.0, "max": 2.0},
            ]
        ),
    )

    captured = _run_display_with_capture(
        monkeypatch,
        metrics_result,
        table_metrics=["NonExistent"],
        plot_metrics=["NonExistent"],
    )
    assert captured == {}


def test_display_metrics_shows_only_tables_when_plot_metrics_empty(monkeypatch) -> None:  # noqa: D103
    alg_a = _AlgorithmStub("A")
    metric_1 = _MetricStub("one")
    metric_2 = _MetricStub("two")

    metrics_result = _build_display_metric_result([alg_a], [metric_1, metric_2], [metric_1, metric_2])

    captured = _run_display_with_capture(
        monkeypatch,
        metrics_result,
        table_metrics=[metric_1],
        plot_metrics=["NonExistent"],
    )

    assert "table" in captured
    assert "plot" not in captured


def test_display_metrics_shows_only_plots_when_table_metrics_empty(monkeypatch) -> None:  # noqa: D103
    alg_a = _AlgorithmStub("A")
    metric_1 = _MetricStub("one")
    metric_2 = _MetricStub("two")

    metrics_result = _build_display_metric_result([alg_a], [metric_1, metric_2], [metric_1, metric_2])

    captured = _run_display_with_capture(
        monkeypatch,
        metrics_result,
        table_metrics=["NonExistent"],
        plot_metrics=[metric_1],
    )

    assert "plot" in captured
    assert "table" not in captured


def test_display_tables_scales_compute_metrics_in_shared_layout(monkeypatch, tmp_path: Path) -> None:  # noqa: D103
    alg_a = _AlgorithmStub("A")
    metric = ml.FunctionCalls()

    metrics_result = MetricResult(
        network_views={alg_a: [_network_metrics_view([_agent_metrics_view(1.0)])]},
        raw_table_results={
            metric: pd.DataFrame({"algorithm": ["A"], "trial": [0], "agent": [0], "value": [20.0]})
        },
        raw_plot_results=None,
        table_results=pd.DataFrame.from_records(
            [{"metric": metric.description, "statistic": "mean", "algorithm": "A", "mean": 20.0, "std": 4.0}]
        ),
        plot_results=None,
    )

    monkeypatch.setattr(pd.DataFrame, "to_latex", lambda self, *args, **kwargs: "latex table stub")
    display_tables(metrics_result, scale_compute=0.5, table_path=tmp_path)

    grid_table = (tmp_path / "table.txt").read_text(encoding="utf-8")

    assert "1.00e+01 ± 2.00e+00" in grid_table


def test_metric_result_available_discovery_properties() -> None:  # noqa: D103
    alg_a = _AlgorithmStub("A")
    alg_b = _AlgorithmStub("B")
    metric_1 = _MetricStub("one")
    metric_2 = _MetricStub("two")

    metrics_result = MetricResult(
        network_views={
            alg_a: [_network_metrics_view([])],
            alg_b: [_network_metrics_view([])],
        },
        raw_table_results={
            metric_1: pd.DataFrame({"algorithm": ["A"], "trial": [0], "agent": [0], "value": [1.0]})
        },
        raw_plot_results={
            metric_2: pd.DataFrame(
                {"algorithm": ["B"], "trial": [0], "agent": [0], "iteration": [0], "value": [2.0]}
            )
        },
        table_results=pd.DataFrame.from_records(
            [{"metric": metric_1.description, "statistic": "avg", "algorithm": "A", "mean": 1.0, "std": 0.1}]
        ),
        plot_results=pd.DataFrame.from_records(
            [{"metric": metric_2.description, "algorithm": "B", "iteration": 0.0, "mean": 2.0, "min": 2.0, "max": 2.0}]
        ),
    )

    assert metrics_result.algorithms == ["A", "B"]
    assert metrics_result.table_metrics == ["one"]
    assert metrics_result.plot_metrics == ["two"]


@pytest.mark.parametrize(
    ("table_metrics", "plot_metrics", "expected_error"),
    [
        ([_MetricStub("same"), _MetricStub("same")], [], "Table metric descriptions must be unique"),
        ([], [_MetricStub("same"), _MetricStub("same")], "Plot metric descriptions must be unique"),
    ],
)
def test_compute_metrics_rejects_duplicate_metric_descriptions(
    monkeypatch,
    table_metrics: list[_MetricStub],
    plot_metrics: list[_MetricStub],
    expected_error: str,
) -> None:  # noqa: D103
    benchmark_result = _build_minimal_benchmark_result()

    with pytest.raises(ValueError, match=expected_error):
        compute_metrics(benchmark_result=benchmark_result, table_metrics=table_metrics, plot_metrics=plot_metrics)


def test_compute_metrics_uses_federated_defaults_and_server_view() -> None:  # noqa: D103
    benchmark_result, algorithm = _build_federated_benchmark_result(iterations=2)

    metrics_result = compute_metrics(benchmark_result=benchmark_result, log_level=40)

    assert metrics_result.network_views is not None
    assert metrics_result.network_views[algorithm][0].server().x_history.max() == 2
    assert metrics_result.raw_table_results is not None
    assert set(metric.description for metric in metrics_result.raw_table_results) == set(metrics_result.table_metrics)
    sample_raw_frame = next(iter(metrics_result.raw_table_results.values()))
    assert list(sample_raw_frame.columns) == ["algorithm", "trial", "agent", "value"]
    table_metric_types = {type(metric) for metric in metrics_result.raw_table_results}
    assert metrics_result.raw_plot_results is not None
    plot_metric_types = {type(metric) for metric in metrics_result.raw_plot_results}
    assert ml.ClientDriftFromServer in table_metric_types
    assert ml.FractionSelectedClients in table_metric_types
    assert ml.ClientDriftFromServer in plot_metric_types

    selected_metric = next(
        metric for metric in metrics_result.raw_table_results if isinstance(metric, ml.FractionSelectedClients)
    )
    sent_messages_metric = next(
        metric for metric in metrics_result.raw_table_results if isinstance(metric, ml.SentMessages)
    )

    assert metrics_result.table_results is not None
    selected_row = metrics_result.table_results[
        (metrics_result.table_results["metric"] == selected_metric.description)
        & (metrics_result.table_results["statistic"] == "")
        & (metrics_result.table_results["algorithm"] == algorithm.name)
    ].iloc[0]
    assert float(selected_row["mean"]) == pytest.approx(1.0)
    assert float(selected_row["std"]) == 0.0
    sent_messages_stats = metrics_result.table_results[
        (metrics_result.table_results["metric"] == sent_messages_metric.description)
        & (metrics_result.table_results["algorithm"] == algorithm.name)
    ]
    assert "sum" not in sent_messages_stats["statistic"].tolist()
    sent_mean_row = sent_messages_stats[sent_messages_stats["statistic"] == "mean"].iloc[0]
    assert float(sent_mean_row["mean"]) == pytest.approx(8.0 / 3.0)
    assert float(sent_mean_row["std"]) == 0.0


def test_compute_metrics_custom_metrics_do_not_append_federated_defaults(monkeypatch) -> None:  # noqa: D103
    benchmark_result, _ = _build_federated_benchmark_result(iterations=1)
    metric = _MetricStub("custom table")
    captured: dict[str, object] = {}

    def _capture_tables(*args, **kwargs):
        captured["table_metrics"] = args[2]
        return {}

    def _capture_plots(*args, **kwargs):
        captured["plot_metrics"] = args[2]
        return {}

    compute_metrics_module = importlib.import_module("decent_bench.benchmark._compute._compute_metrics")
    monkeypatch.setattr(compute_metrics_module, "compute_table_metrics", _capture_tables)
    monkeypatch.setattr(compute_metrics_module, "compute_plot_metrics", _capture_plots)

    compute_metrics(benchmark_result=benchmark_result, table_metrics=[metric], plot_metrics=[])

    assert [m.description for m in captured["table_metrics"]] == [metric.description]
    assert captured["plot_metrics"] == []


# -----------------------------------------------------------------------------
# compute_plots Truncation Tests
# -----------------------------------------------------------------------------


def test_compute_plots_truncates_trials_at_first_non_finite_value() -> None:  # noqa: D103
    alg_a = _AlgorithmStub("A")
    metric = _PlotMetricStub(
        "diverging plot",
        {
            1.0: [(0.0, 1.0), (1.0, 2.0), (2.0, np.nan)],
            2.0: [(0.0, 3.0), (1.0, 4.0), (2.0, 5.0)],
        },
    )

    raw_plot_results = compute_plot_metrics(
        {
            alg_a: [
                _network_metrics_view([_agent_metrics_view(1.0)]),
                _network_metrics_view([_agent_metrics_view(2.0)]),
            ]
        },
        SimpleNamespace(),
        [metric],
        [0, 1, 2],
    )
    plot_results = aggregate_plot_metrics(raw_plot_results)

    assert plot_results is not None
    metric_df = plot_results[(plot_results["metric"] == metric.description) & (plot_results["algorithm"] == alg_a.name)]
    assert metric_df["iteration"].tolist() == [0.0, 1.0, 2.0]
    assert metric_df["mean"].tolist() == [2.0, 3.0, np.inf]
    assert metric_df["min"].tolist() == [1.0, 2.0, 5.0]
    assert metric_df["max"].tolist() == [3.0, 4.0, np.inf]


def test_compute_plots_drops_trials_without_finite_prefix() -> None:  # noqa: D103
    alg_a = _AlgorithmStub("A")
    metric = _PlotMetricStub(
        "partially diverging plot",
        {
            1.0: [(0.0, np.nan), (1.0, 2.0)],
            2.0: [(0.0, 3.0), (1.0, 4.0)],
        },
    )

    raw_plot_results = compute_plot_metrics(
        {
            alg_a: [
                _network_metrics_view([_agent_metrics_view(1.0)]),
                _network_metrics_view([_agent_metrics_view(2.0)]),
            ]
        },
        SimpleNamespace(),
        [metric],
        [0, 1],
    )
    plot_results = aggregate_plot_metrics(raw_plot_results)

    assert plot_results is not None
    metric_df = plot_results[(plot_results["metric"] == metric.description) & (plot_results["algorithm"] == alg_a.name)]
    assert metric_df["iteration"].tolist() == [0.0, 1.0]
    assert metric_df["mean"].tolist() == [np.inf, 3.0]
    assert metric_df["min"].tolist() == [3.0, 2.0]
    assert metric_df["max"].tolist() == [np.inf, 4.0]


def test_compute_plots_omits_metric_without_any_finite_prefix() -> None:  # noqa: D103
    alg_a = _AlgorithmStub("A")
    metric = _PlotMetricStub(
        "invalid plot",
        {
            1.0: [(0.0, np.nan)],
            2.0: [(0.0, np.inf)],
        },
    )

    raw_plot_results = compute_plot_metrics(
        {
            alg_a: [
                _network_metrics_view([_agent_metrics_view(1.0)]),
                _network_metrics_view([_agent_metrics_view(2.0)]),
            ]
        },
        SimpleNamespace(),
        [metric],
        [0],
    )
    plot_results = aggregate_plot_metrics(raw_plot_results)

    assert plot_results is not None
    assert plot_results["mean"].tolist() == [np.inf]
    assert plot_results["min"].tolist() == [np.inf]
    assert plot_results["max"].tolist() == [np.inf]


def test_compute_plots_truncates_trials_at_over_threshold_value() -> None:  # noqa: D103
    alg_a = _AlgorithmStub("A")
    metric = _PlotMetricStub(
        "over-threshold plot",
        {
            1.0: [(0.0, 1.0), (1.0, 2.0), (2.0, MAX_ABS_METRIC_VALUE * 10)],
            2.0: [(0.0, 3.0), (1.0, 4.0), (2.0, 5.0)],
        },
    )

    raw_plot_results = compute_plot_metrics(
        {
            alg_a: [
                _network_metrics_view([_agent_metrics_view(1.0)]),
                _network_metrics_view([_agent_metrics_view(2.0)]),
            ]
        },
        SimpleNamespace(),
        [metric],
        [0, 1, 2],
    )
    plot_results = aggregate_plot_metrics(raw_plot_results)

    assert plot_results is not None
    metric_df = plot_results[(plot_results["metric"] == metric.description) & (plot_results["algorithm"] == alg_a.name)]
    assert metric_df["iteration"].tolist() == [0.0, 1.0, 2.0]
    assert metric_df["mean"].tolist() == [2.0, 3.0, np.inf]
    assert metric_df["min"].tolist() == [1.0, 2.0, 5.0]
    assert metric_df["max"].tolist() == [3.0, 4.0, np.inf]


# -----------------------------------------------------------------------------
# Legend Layout Threshold Tests
# -----------------------------------------------------------------------------


def test_select_legend_mode_prefers_same_figure_for_small_label_sets() -> None:  # noqa: D103
    mode, cols, rows = _select_legend_mode(["A", "B", "C", "D"])

    assert mode == "same-figure"
    assert cols == 3
    assert rows == 2


def test_select_legend_mode_prefers_same_figure_for_medium_label_sets() -> None:  # noqa: D103
    labels = [f"alg_{i}" for i in range(8)]
    mode, cols, rows = _select_legend_mode(labels)

    assert mode == "same-figure"
    assert cols == 3
    assert rows == 3


def test_select_legend_mode_uses_separate_for_many_labels() -> None:  # noqa: D103
    labels = [f"alg_{i}" for i in range(9)]
    mode, cols, rows = _select_legend_mode(labels)

    assert mode == "separate"
    assert cols == 3
    assert rows == 3


def test_select_legend_mode_uses_separate_for_long_labels() -> None:  # noqa: D103
    long_label = "algorithm_with_a_very_long_name"
    labels = [f"{long_label}_{i}" for i in range(6)]
    mode, cols, rows = _select_legend_mode(labels)

    assert mode == "separate"
    assert cols == 3
    assert rows == 2


def test_select_legend_mode_uses_separate_for_many_estimated_rows() -> None:  # noqa: D103
    labels = [f"alg_{i}" for i in range(16)]
    mode, cols, rows = _select_legend_mode(labels)

    assert mode == "separate"
    assert cols == 3
    assert rows == 6


def test_get_separate_legend_path_appends_legend_suffix() -> None:  # noqa: D103
    plot_path = Path("plots/plot_fig1.png")

    legend_path = _get_separate_legend_path(plot_path)

    assert legend_path.as_posix() == "plots/plot_fig1_legend.png"


def _test_figure(*, figsize: tuple[float, float] | None = None) -> Figure:
    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)
    return fig


def test_create_separate_legend_figure_tightly_fits_legend(monkeypatch) -> None:  # noqa: D103
    monkeypatch.setattr(
        display_plots_module.plt,
        "figure",
        lambda *args, **kwargs: _test_figure(figsize=kwargs.get("figsize")),
    )
    handles = [
        Line2D([], [], label="Algorithm A"),
        Line2D([], [], label="Algorithm B"),
    ]
    labels = ["Algorithm A", "Algorithm B"]

    legend_fig = _create_separate_legend_figure(handles, labels, label_cols=2, estimated_rows=1)
    try:
        legend = legend_fig.legends[0]
        legend_fig.canvas.draw()
        renderer = legend_fig.canvas.get_renderer()
        legend_bbox = legend.get_window_extent(renderer)
        figure_width, figure_height = legend_fig.get_size_inches()
        legend_width = legend_bbox.width / legend_fig.dpi
        legend_height = legend_bbox.height / legend_fig.dpi

        assert figure_width >= legend_width
        assert figure_height >= legend_height
        assert figure_width - legend_width < 0.5
        assert figure_height - legend_height < 0.5
    finally:
        plt.close(legend_fig)


def test_add_legend_and_save_creates_missing_parent_directory(monkeypatch) -> None:  # noqa: D103
    fig = _test_figure()
    ax = fig.subplots()
    ax.plot([0.0, 1.0], [0.0, 1.0], label="A")
    target = Path("missing_parent") / "nested" / "results" / "plot.png"
    mkdir_calls = []
    savefig_calls = []

    def _capture_mkdir(self, *, parents=False, exist_ok=False):  # noqa: ANN001, ANN202, FBT002
        mkdir_calls.append((self, parents, exist_ok))

    def _capture_savefig(path, *args, **kwargs):  # noqa: ANN001, ANN202, ARG001
        savefig_calls.append(path)

    monkeypatch.setattr(Path, "mkdir", _capture_mkdir)
    monkeypatch.setattr(fig, "savefig", _capture_savefig)

    try:
        _add_legend_and_save(fig, [ax], plot_path=target)

        assert mkdir_calls == [(target.parent, True, True)]
        assert savefig_calls == [target]
    finally:
        plt.close(fig)


# -----------------------------------------------------------------------------
# Metric Availability Tests
# -----------------------------------------------------------------------------


def test_xerror_unavailable_without_x_optimal() -> None:  # noqa: D103
    x_error = XError([np.average])
    available, reason = x_error.is_available(SimpleNamespace(x_optimal=None))
    assert not available
    assert reason == "requires problem.x_optimal"


def test_xerror_available_with_x_optimal() -> None:  # noqa: D103
    x_error = XError([np.average])
    available, reason = x_error.is_available(SimpleNamespace(x_optimal=np.array([0.0])))
    assert available
    assert reason is None


def test_regret_unavailable_without_x_optimal() -> None:  # noqa: D103
    regret = Regret([np.average])
    available, reason = regret.is_available(SimpleNamespace(x_optimal=None))
    assert not available
    assert reason == "requires problem.x_optimal"


def test_metrics_unavailable_without_test_data() -> None:  # noqa: D103
    problem = SimpleNamespace(test_data=None)
    metrics = [
        Accuracy([np.average]),
        BalancedAccuracy([np.average]),
        MSE([np.average]),
        Precision([np.average]),
        Recall([np.average]),
    ]

    for metric in metrics:
        available, reason = metric.is_available(problem)
        assert not available
        assert reason == "requires problem.test_data"


def test_metrics_unavailable_without_empirical_risk_cost() -> None:  # noqa: D103
    network = SimpleNamespace(agents=lambda: [SimpleNamespace(cost=object())])
    problem = SimpleNamespace(test_data=[(np.array([1.0]), 0)], network=network)
    metrics = [
        Accuracy([np.average]),
        BalancedAccuracy([np.average]),
        MSE([np.average]),
        Precision([np.average]),
        Recall([np.average]),
    ]

    for metric in metrics:
        available, reason = metric.is_available(problem)
        assert not available
        assert "EmpiricalRiskCost" in reason


def test_classification_metrics_unavailable_with_float_targets() -> None:  # noqa: D103
    lr_cost = LinearRegressionCost([(np.array([1.0]), np.array([1.0]))])
    network = SimpleNamespace(agents=lambda: [SimpleNamespace(cost=lr_cost)])
    problem = SimpleNamespace(test_data=[(np.array([0.0]), 0.1)], network=network)

    for metric in [
        Accuracy([np.average]),
        BalancedAccuracy([np.average]),
        Precision([np.average]),
        Recall([np.average]),
    ]:
        available, reason = metric.is_available(problem)
        assert not available
        assert "integer targets" in reason


def test_balanced_accuracy_uses_per_class_recall() -> None:  # noqa: D103
    labels = [0, 0, 1, 1, 2, 2]
    predictions = [0, 1, 1, 1, 2, 0]
    problem = _ProblemStub(labels)
    agent = AgentMetricsView(
        id=uuid4(),
        cost=_PredictCostStub(predictions),
        x_history={0: np.array([0.0])},
        n_x_updates=0,
        n_function_calls=0.0,
        n_gradient_calls=0.0,
        n_hessian_calls=0.0,
        n_proximal_calls=0.0,
        n_sent_messages=0,
        n_received_messages=0,
        n_sent_messages_dropped=0,
        n_times_selected=0,
    )
    network_view = _network_metrics_view([agent])

    result = ml.BalancedAccuracy().compute(network_view, problem, 0)

    assert result == [balanced_accuracy_score(labels, predictions)]


def test_balanced_accuracy_metrics_are_not_registered_by_default() -> None:  # noqa: D103
    default_table_descriptions = {metric.description for metric in ml._DEFAULT_TABLE_METRICS}  # noqa: SLF001
    default_plot_descriptions = {metric.description for metric in ml._DEFAULT_PLOT_METRICS}  # noqa: SLF001

    assert ml.BalancedAccuracy().description == "balanced accuracy"
    assert ml.ServerBalancedAccuracy().description == "server balanced accuracy"
    assert "balanced accuracy" not in default_table_descriptions
    assert "balanced accuracy" not in default_plot_descriptions
    assert "server balanced accuracy" not in default_table_descriptions
    assert "server balanced accuracy" not in default_plot_descriptions


def test_server_mse_availability_and_values() -> None:  # noqa: D103
    test_data = [(np.array([1.0]), np.array([1.0]))]
    cost = LinearRegressionCost(test_data)
    client = Agent(cost)
    client.initialize(x=np.array([0.0]))

    unavailable_problem = SimpleNamespace(
        test_data=test_data,
        network=SimpleNamespace(agents=lambda: [client]),
    )
    available, reason = ml.ServerMSE([np.average]).is_available(unavailable_problem)
    assert not available
    assert "FedNetwork" in reason

    client = Agent(cost)
    client.initialize(x=np.array([0.0]))

    fed_problem_without_test_data = BenchmarkProblem(network=FedNetwork([client]))
    available, reason = ml.ServerMSE([np.average]).is_available(fed_problem_without_test_data)
    assert not available
    assert reason == "requires problem.test_data"

    client = Agent(cost)
    client.initialize(x=np.array([0.0]))

    problem = BenchmarkProblem(network=FedNetwork([client]), test_data=test_data)
    metric = ml.ServerMSE([np.average])
    available, reason = metric.is_available(problem)
    assert available
    assert reason is None

    client_view = AgentMetricsView.from_agent(client)
    server_history = AgentHistory()
    server_history[0] = np.array([0.0])
    server_history[1] = np.array([1.0])
    server_view = AgentMetricsView(
        id=uuid4(),
        cost=cost,
        x_history=server_history,
        n_x_updates=0,
        n_function_calls=0.0,
        n_gradient_calls=0.0,
        n_hessian_calls=0.0,
        n_proximal_calls=0.0,
        n_sent_messages=0,
        n_received_messages=0,
        n_sent_messages_dropped=0,
        n_times_selected=0,
    )

    network_view = _network_metrics_view(
        [client_view, server_view],
        network_type=NetworkType.FEDERATED,
        server=server_view,
    )
    assert metric.compute(network_view, problem, 0) == [1.0]
    assert metric.compute(network_view, problem, 1) == [0.0]


def test_server_accuracy_availability_and_values() -> None:  # noqa: D103
    train_data = [(np.array([1.0]), np.array([1])), (np.array([-1.0]), np.array([0]))]
    test_data = [(np.array([1.0]), 1), (np.array([-1.0]), 0)]
    cost = LogisticRegressionCost(train_data)
    client = Agent(cost)
    client.initialize(x=np.array([0.0]))
    problem = BenchmarkProblem(network=FedNetwork([client]), test_data=test_data)

    client_for_float_targets = Agent(cost)
    client_for_float_targets.initialize(x=np.array([0.0]))

    float_target_problem = BenchmarkProblem(
        network=FedNetwork([client_for_float_targets]),
        test_data=[(np.array([1.0]), 1.0), (np.array([-1.0]), 0.0)],
    )
    metric = ml.ServerAccuracy([np.average])
    available, reason = metric.is_available(float_target_problem)
    assert not available
    assert "integer targets" in reason

    available, reason = metric.is_available(problem)
    assert available
    assert reason is None

    client_view = AgentMetricsView.from_agent(client)
    server_history = AgentHistory()
    server_history[0] = np.array([0.0])
    server_history[1] = np.array([10.0])
    server_view = AgentMetricsView(
        id=uuid4(),
        cost=cost,
        x_history=server_history,
        n_x_updates=0,
        n_function_calls=0.0,
        n_gradient_calls=0.0,
        n_hessian_calls=0.0,
        n_proximal_calls=0.0,
        n_sent_messages=0,
        n_received_messages=0,
        n_sent_messages_dropped=0,
        n_times_selected=0,
    )

    network_view = _network_metrics_view(
        [client_view, server_view],
        network_type=NetworkType.FEDERATED,
        server=server_view,
    )
    assert metric.compute(network_view, problem, 0) == [0.5]
    assert metric.compute(network_view, problem, 1) == [1.0]


def test_server_balanced_accuracy_availability_and_values() -> None:  # noqa: D103
    train_data = [(np.array([1.0]), np.array([1])), (np.array([-1.0]), np.array([0]))]
    test_data = [(np.array([1.0]), 1), (np.array([-1.0]), 0)]
    cost = LogisticRegressionCost(train_data)
    client = Agent(cost)
    client.initialize(x=np.array([0.0]))
    problem = BenchmarkProblem(network=FedNetwork([client]), test_data=test_data)

    client_for_float_targets = Agent(cost)
    client_for_float_targets.initialize(x=np.array([0.0]))

    float_target_problem = BenchmarkProblem(
        network=FedNetwork([client_for_float_targets]),
        test_data=[(np.array([1.0]), 1.0), (np.array([-1.0]), 0.0)],
    )
    metric = ml.ServerBalancedAccuracy([np.average])
    available, reason = metric.is_available(float_target_problem)
    assert not available
    assert "integer targets" in reason

    available, reason = metric.is_available(problem)
    assert available
    assert reason is None

    labels = [0, 0, 1, 1, 2, 2]
    predictions = [0, 1, 1, 1, 2, 0]
    stub_problem = _ProblemStub(labels)
    client_view = AgentMetricsView(
        id=uuid4(),
        cost=_PredictCostStub(predictions),
        x_history={0: np.array([0.0])},
        n_x_updates=0,
        n_function_calls=0.0,
        n_gradient_calls=0.0,
        n_hessian_calls=0.0,
        n_proximal_calls=0.0,
        n_sent_messages=0,
        n_received_messages=0,
        n_sent_messages_dropped=0,
        n_times_selected=0,
    )
    server_view = AgentMetricsView(
        id=uuid4(),
        cost=_PredictCostStub([]),
        x_history={0: np.array([0.0])},
        n_x_updates=0,
        n_function_calls=0.0,
        n_gradient_calls=0.0,
        n_hessian_calls=0.0,
        n_proximal_calls=0.0,
        n_sent_messages=0,
        n_received_messages=0,
        n_sent_messages_dropped=0,
        n_times_selected=0,
    )
    network_view = _network_metrics_view(
        [client_view, server_view],
        network_type=NetworkType.FEDERATED,
        server=server_view,
    )

    assert metric.compute(network_view, stub_problem, 0) == [balanced_accuracy_score(labels, predictions)]


def test_is_available_default_returns_true() -> None:  # noqa: D103
    """Base Metric.is_available default: always available."""
    metric = _MetricStub("t")
    available, reason = metric.is_available(SimpleNamespace())
    assert available
    assert reason is None


# -----------------------------------------------------------------------------
# Plot Metric Stub
# -----------------------------------------------------------------------------


class _PlotMetricStub(Metric):
    def __init__(self, description: str, plot_data_by_x_value: dict[float, list[tuple[float, float]]]) -> None:
        super().__init__(fmt=".2e", y_log=False)
        self._description = description
        self._plot_data_by_x_value = plot_data_by_x_value

    @property
    def description(self) -> str:
        return self._description

    def compute(self, network, problem, iteration):  # noqa: D102
        first_agent = network.agents()[0]
        last_iteration = first_agent.x_history.max()
        x_value = float(first_agent.x_history[last_iteration][0])
        data_by_iter = dict(self._plot_data_by_x_value[x_value])
        return [float(data_by_iter[float(iteration)])]

    def get_plot_data(self, network, problem):  # noqa: D102
        first_agent = network.agents()[0]
        last_iteration = first_agent.x_history.max()
        x_value = float(first_agent.x_history[last_iteration][0])
        return self._plot_data_by_x_value[x_value]
