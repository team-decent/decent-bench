import matplotlib.pyplot as plt
import numpy as np
import pytest
from copy import deepcopy
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from decent_bench.agents import Agent, AgentHistory
from decent_bench.algorithms.federated import FedAvg
from decent_bench.benchmark import BenchmarkProblem, BenchmarkResult, compute_metrics
from decent_bench.benchmark._display import display_metrics
from decent_bench.benchmark._metric_result import MetricResult
from decent_bench.costs import LinearRegressionCost, LogisticRegressionCost, QuadraticCost
from decent_bench.metrics._metric import Metric
from decent_bench.metrics._metrics_view import AgentMetricsView
from decent_bench.metrics import metric_library as ml
from decent_bench.metrics._plots import (
    MAX_Y_PLOT_VALUE,
    _add_legend_and_save,
    _create_separate_legend_figure,
    _get_separate_legend_path,
    _select_legend_mode,
    compute_plots,
)
from decent_bench.metrics.metric_library import Accuracy, MSE, Precision, Recall, Regret, XError
from decent_bench.networks import FedNetwork


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
        super().__init__([np.average], fmt=".2e", y_log=False)
        self._description = description

    @property
    def description(self) -> str:
        return self._description

    def get_data_from_trial(self, agents, problem, iteration):  # noqa: D102
        return [0.0]


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
        is_server=False,
    )


def _run_display_with_capture(
    monkeypatch,
    metrics_result: MetricResult,
    **display_kwargs,
) -> dict[str, MetricResult]:
    captured: dict[str, MetricResult] = {}

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

    monkeypatch.setattr("decent_bench.benchmark._display.display_tables", _capture_tables)
    monkeypatch.setattr("decent_bench.benchmark._display.display_plots", _capture_plots)

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
    table_results: dict[_AlgorithmStub, dict[_MetricStub, dict[str, tuple[float, float]]]] | None = None,
    plot_results: dict[_AlgorithmStub, dict[_MetricStub, tuple[list[float], list[float], list[float], list[float]]]]
    | None = None,
) -> MetricResult:
    if agent_x_values is None:
        agent_x_values = [1.0] * len(algorithms)

    default_table_results: dict[_AlgorithmStub, dict[_MetricStub, dict[str, tuple[float, float]]]] = {}
    default_plot_results: dict[
        _AlgorithmStub, dict[_MetricStub, tuple[list[float], list[float], list[float], list[float]]]
    ] = {}

    for alg_idx, alg in enumerate(algorithms):
        default_table_results[alg] = {}
        default_plot_results[alg] = {}
        for metric_idx, metric in enumerate(table_metrics):
            value = float(alg_idx + metric_idx + 1)
            default_table_results[alg][metric] = {"avg": (value, 0.0)}
        for metric_idx, metric in enumerate(plot_metrics):
            value = float(alg_idx + metric_idx + 1)
            default_plot_results[alg][metric] = ([0.0], [value], [value], [value])

    return MetricResult(
        agent_metrics={alg: [[_agent_metrics_view(agent_x_values[idx])]] for idx, alg in enumerate(algorithms)},
        table_metrics=table_metrics,
        plot_metrics=plot_metrics,
        table_results=table_results or default_table_results,
        plot_results=plot_results or default_plot_results,
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
    assert [alg.name for alg in captured["table"].table_results] == expected_algorithms
    assert [alg.name for alg in captured["table"].plot_results] == expected_algorithms
    assert [alg.name for alg in captured["table"].agent_metrics] == expected_algorithms
    assert [alg.name for alg in metrics_result.table_results] == ["A", "B"]


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
        table_results={
            alg_a: {
                valid_metric: {"avg": (1.0, 0.1)},
                nan_metric: {"avg": (np.nan, np.nan)},
            },
            alg_b: {
                valid_metric: {"avg": (2.0, 0.1)},
                nan_metric: {"avg": (np.nan, np.nan)},
            },
        },
        plot_results={
            alg_a: {valid_metric: ([0.0], [1.0], [1.0], [1.0])},
            alg_b: {valid_metric: ([0.0], [2.0], [2.0], [2.0])},
        },
    )

    captured = _run_display_with_capture(monkeypatch, metrics_result)

    assert "table" in captured
    assert "plot" in captured
    assert [metric.description for metric in captured["table"].table_metrics] == ["valid", "nan"]
    assert [metric.description for metric in captured["table"].plot_metrics] == ["valid", "nan"]
    for algorithm_results in captured["table"].table_results.values():
        assert [metric.description for metric in algorithm_results] == ["valid", "nan"]


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
    assert [metric.description for metric in captured["table"].table_metrics] == ["two"]
    assert [metric.description for metric in captured["table"].plot_metrics] == ["one"]


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
    assert [metric.description for metric in captured["table"].table_metrics] == ["one", "two"]
    assert [metric.description for metric in captured["table"].plot_metrics] == ["one", "two"]


def test_display_metrics_filters_algorithms_with_mixed_objects_and_names(monkeypatch) -> None:  # noqa: D103
    alg_a = _AlgorithmStub("A")
    alg_b = _AlgorithmStub("B")
    alg_c = _AlgorithmStub("C")
    metric = _MetricStub("one")

    metrics_result = _build_display_metric_result([alg_a, alg_b, alg_c], [metric], [metric])

    captured = _run_display_with_capture(monkeypatch, metrics_result, algorithms=[alg_a, "C"])

    assert "table" in captured
    assert [alg.name for alg in captured["table"].table_results] == ["A", "C"]
    assert [alg.name for alg in captured["table"].plot_results] == ["A", "C"]
    assert [alg.name for alg in captured["table"].agent_metrics] == ["A", "C"]
    assert [alg.name for alg in metrics_result.table_results] == ["A", "B", "C"]


def test_display_metrics_raises_when_all_algorithms_filtered_out(monkeypatch) -> None:  # noqa: D103
    alg_a = _AlgorithmStub("A")
    metric = _MetricStub("one")

    metrics_result = MetricResult(
        agent_metrics={alg_a: [[_agent_metrics_view(1.0)]]},
        table_metrics=[metric],
        plot_metrics=[metric],
        table_results={alg_a: {metric: {"avg": (1.0, 0.0)}}},
        plot_results={alg_a: {metric: ([0.0], [1.0], [1.0], [1.0])}},
    )

    with pytest.raises(ValueError, match="No algorithms remain after filtering"):
        display_metrics(metrics_result=metrics_result, algorithms=["NonExistent"])


def test_display_metrics_raises_when_all_table_and_plot_metrics_filtered_out(monkeypatch) -> None:  # noqa: D103
    alg_a = _AlgorithmStub("A")
    metric_1 = _MetricStub("one")
    metric_2 = _MetricStub("two")

    metrics_result = MetricResult(
        agent_metrics={alg_a: [[_agent_metrics_view(1.0)]]},
        table_metrics=[metric_1, metric_2],
        plot_metrics=[metric_1, metric_2],
        table_results={
            alg_a: {
                metric_1: {"avg": (1.0, 0.0)},
                metric_2: {"avg": (2.0, 0.0)},
            }
        },
        plot_results={
            alg_a: {
                metric_1: ([0.0], [1.0], [1.0], [1.0]),
                metric_2: ([0.0], [2.0], [2.0], [2.0]),
            }
        },
    )

    with pytest.raises(ValueError, match="No table or plot metrics remain after filtering"):
        display_metrics(
            metrics_result=metrics_result,
            table_metrics=["NonExistent"],
            plot_metrics=["NonExistent"],
        )


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


def test_metric_result_available_discovery_properties() -> None:  # noqa: D103
    alg_a = _AlgorithmStub("A")
    alg_b = _AlgorithmStub("B")
    metric_1 = _MetricStub("one")
    metric_2 = _MetricStub("two")

    metrics_result = MetricResult(
        agent_metrics={alg_a: [[[]]], alg_b: [[[]]]},
        table_metrics=[metric_1, metric_1, metric_2],
        plot_metrics=[metric_1, metric_1, metric_2],
        table_results={alg_a: {metric_1: {"avg": (1.0, 0.1)}}},
        plot_results={alg_b: {metric_2: ([0.0], [2.0], [2.0], [2.0])}},
    )

    assert metrics_result.available_algorithms == ["A", "B"]
    assert metrics_result.available_table_metrics == ["one", "two"]
    assert metrics_result.available_plot_metrics == ["one", "two"]


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

    monkeypatch.setattr("decent_bench.benchmark._compute.compute_tables", lambda *args, **kwargs: {})
    monkeypatch.setattr("decent_bench.benchmark._compute.compute_plots", lambda *args, **kwargs: {})

    with pytest.raises(ValueError, match=expected_error):
        compute_metrics(benchmark_result=benchmark_result, table_metrics=table_metrics, plot_metrics=plot_metrics)


def test_compute_metrics_uses_federated_defaults_and_server_view() -> None:  # noqa: D103
    benchmark_result, algorithm = _build_federated_benchmark_result(iterations=2)

    metrics_result = compute_metrics(benchmark_result=benchmark_result, log_level=40)

    assert metrics_result.agent_metrics is not None
    server_views = [view for view in metrics_result.agent_metrics[algorithm][0] if view.is_server]
    assert len(server_views) == 1
    assert server_views[0].x_history.max() == 2
    table_metric_types = {type(metric) for metric in metrics_result.table_metrics or []}
    plot_metric_types = {type(metric) for metric in metrics_result.plot_metrics or []}
    assert ml.ClientDriftFromServer in table_metric_types
    assert ml.FractionSelectedClients in table_metric_types
    assert ml.ClientDriftFromServer in plot_metric_types

    selected_metric = next(
        metric for metric in metrics_result.table_metrics or [] if isinstance(metric, ml.FractionSelectedClients)
    )
    sent_messages_metric = next(
        metric for metric in metrics_result.table_metrics or [] if isinstance(metric, ml.SentMessages)
    )

    assert metrics_result.table_results is not None
    assert metrics_result.table_results[algorithm][selected_metric][""] == (1.0, 0.0)
    assert metrics_result.table_results[algorithm][sent_messages_metric]["sum"] == (8.0, 0.0)


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

    monkeypatch.setattr("decent_bench.benchmark._compute.compute_tables", _capture_tables)
    monkeypatch.setattr("decent_bench.benchmark._compute.compute_plots", _capture_plots)

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

    plot_results = compute_plots(
        {alg_a: [[_agent_metrics_view(1.0)], [_agent_metrics_view(2.0)]]},
        SimpleNamespace(),
        [metric],
    )

    assert metric in plot_results[alg_a]
    x, y_mean, y_min, y_max = plot_results[alg_a][metric]
    assert list(x) == [0.0, 1.0]
    assert list(y_mean) == [2.0, 3.0]
    assert list(y_min) == [1.0, 2.0]
    assert list(y_max) == [3.0, 4.0]


def test_compute_plots_drops_trials_without_finite_prefix() -> None:  # noqa: D103
    alg_a = _AlgorithmStub("A")
    metric = _PlotMetricStub(
        "partially diverging plot",
        {
            1.0: [(0.0, np.nan), (1.0, 2.0)],
            2.0: [(0.0, 3.0), (1.0, 4.0)],
        },
    )

    plot_results = compute_plots(
        {alg_a: [[_agent_metrics_view(1.0)], [_agent_metrics_view(2.0)]]},
        SimpleNamespace(),
        [metric],
    )

    assert metric in plot_results[alg_a]
    x, y_mean, y_min, y_max = plot_results[alg_a][metric]
    assert list(x) == [0.0, 1.0]
    assert list(y_mean) == [3.0, 4.0]
    assert list(y_min) == [3.0, 4.0]
    assert list(y_max) == [3.0, 4.0]


def test_compute_plots_omits_metric_without_any_finite_prefix() -> None:  # noqa: D103
    alg_a = _AlgorithmStub("A")
    metric = _PlotMetricStub(
        "invalid plot",
        {
            1.0: [(0.0, np.nan)],
            2.0: [(0.0, np.inf)],
        },
    )

    plot_results = compute_plots(
        {alg_a: [[_agent_metrics_view(1.0)], [_agent_metrics_view(2.0)]]},
        SimpleNamespace(),
        [metric],
    )

    assert metric not in plot_results[alg_a]


def test_compute_plots_truncates_trials_at_over_threshold_value() -> None:  # noqa: D103
    alg_a = _AlgorithmStub("A")
    metric = _PlotMetricStub(
        "over-threshold plot",
        {
            1.0: [(0.0, 1.0), (1.0, 2.0), (2.0, MAX_Y_PLOT_VALUE * 10)],
            2.0: [(0.0, 3.0), (1.0, 4.0), (2.0, 5.0)],
        },
    )

    plot_results = compute_plots(
        {alg_a: [[_agent_metrics_view(1.0)], [_agent_metrics_view(2.0)]]},
        SimpleNamespace(),
        [metric],
    )

    assert metric in plot_results[alg_a]
    x, y_mean, y_min, y_max = plot_results[alg_a][metric]
    assert list(x) == [0.0, 1.0]
    assert list(y_mean) == [2.0, 3.0]
    assert list(y_min) == [1.0, 2.0]
    assert list(y_max) == [3.0, 4.0]


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
        "decent_bench.metrics._plots.plt.figure",
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
    metrics = [Accuracy([np.average]), MSE([np.average]), Precision([np.average]), Recall([np.average])]

    for metric in metrics:
        available, reason = metric.is_available(problem)
        assert not available
        assert reason == "requires problem.test_data"


def test_metrics_unavailable_without_empirical_risk_cost() -> None:  # noqa: D103
    network = SimpleNamespace(agents=lambda: [SimpleNamespace(cost=object())])
    problem = SimpleNamespace(test_data=[(np.array([1.0]), 0)], network=network)
    metrics = [Accuracy([np.average]), MSE([np.average]), Precision([np.average]), Recall([np.average])]

    for metric in metrics:
        available, reason = metric.is_available(problem)
        assert not available
        assert "EmpiricalRiskCost" in reason


def test_classification_metrics_unavailable_with_float_targets() -> None:  # noqa: D103
    lr_cost = LinearRegressionCost([(np.array([1.0]), np.array([1.0]))])
    network = SimpleNamespace(agents=lambda: [SimpleNamespace(cost=lr_cost)])
    problem = SimpleNamespace(test_data=[(np.array([0.0]), 0.1)], network=network)

    for metric in [Accuracy([np.average]), Precision([np.average]), Recall([np.average])]:
        available, reason = metric.is_available(problem)
        assert not available
        assert "integer targets" in reason


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
        is_server=True,
    )

    assert metric.get_table_data([client_view, server_view], problem) == (0.0,)
    assert metric.get_plot_data([client_view, server_view], problem) == [(0, 1.0), (1, 0.0)]


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
        is_server=True,
    )

    assert metric.get_table_data([client_view, server_view], problem) == (1.0,)
    assert metric.get_plot_data([client_view, server_view], problem) == [(0, 0.5), (1, 1.0)]


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
        super().__init__([np.average], fmt=".2e", y_log=False)
        self._description = description
        self._plot_data_by_x_value = plot_data_by_x_value

    @property
    def description(self) -> str:
        return self._description

    def get_data_from_trial(self, agents, problem, iteration):  # noqa: D102
        return [0.0]

    def get_plot_data(self, agents, problem):  # noqa: D102
        last_iteration = agents[0].x_history.max()
        x_value = float(agents[0].x_history[last_iteration][0])
        return self._plot_data_by_x_value[x_value]
