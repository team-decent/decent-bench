import matplotlib.pyplot as plt
import numpy as np
import pytest
import tempfile
from pathlib import Path
from types import SimpleNamespace

from decent_bench.agents import AgentHistory, AgentMetricsView
from decent_bench.benchmark import BenchmarkProblem, BenchmarkResult, compute_metrics
from decent_bench.benchmark._display import display_metrics
from decent_bench.benchmark._metric_result import MetricResult
from decent_bench.costs import LinearRegressionCost
from decent_bench.metrics._metric import Metric
from decent_bench.metrics._plots import (
    MAX_Y_PLOT_VALUE,
    _add_legend_and_save,
    _create_separate_legend_figure,
    _get_separate_legend_path,
    _select_legend_mode,
    compute_plots,
)
from decent_bench.metrics.metric_library import Accuracy, MSE, Precision, Recall, Regret, XError


# -----------------------------------------------------------------------------
# Test Helpers
# -----------------------------------------------------------------------------

class _AlgorithmStub:
    def __init__(self, name: str) -> None:
        self.name = name


class _MetricStub(Metric):
    def __init__(
        self,
        table_description: str,
        plot_description: str,
    ) -> None:
        super().__init__([np.average], fmt=".2e", y_log=False)
        self._table_description = table_description
        self._plot_description = plot_description

    @property
    def table_description(self) -> str:
        return self._table_description

    @property
    def plot_description(self) -> str:
        return self._plot_description

    def get_data_from_trial(self, agents, problem, iteration):  # noqa: D102
        return [0.0]


def _agent_metrics_view(x_value: float) -> AgentMetricsView:
    history = AgentHistory()
    history[0] = np.array([x_value])
    return AgentMetricsView(
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
    )


def _run_display_with_capture(
    monkeypatch,
    metrics_result: MetricResult,
    **display_kwargs,
) -> dict[str, MetricResult]:
    captured: dict[str, MetricResult] = {}

    def _capture_tables(
        passed_metrics_result: MetricResult,
        table_fmt: str = "grid",
        scale_compute: float = 1.0,
        table_path=None,
    ) -> None:
        captured["table"] = passed_metrics_result

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
    ) -> None:
        captured["plot"] = passed_metrics_result

    monkeypatch.setattr("decent_bench.benchmark._display.display_tables", _capture_tables)
    monkeypatch.setattr("decent_bench.benchmark._display.display_plots", _capture_plots)

    display_metrics(metrics_result=metrics_result, **display_kwargs)
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
    )
    fake_network = SimpleNamespace(agents=lambda: [fake_agent])
    algorithm = _AlgorithmStub("A")
    return BenchmarkResult(
        problem=BenchmarkProblem(network=fake_network),
        states={algorithm: [fake_network]},
    )


def _build_display_metric_result(
    algorithms: list[_AlgorithmStub],
    table_metrics: list[_MetricStub],
    plot_metrics: list[_MetricStub] | list[list[_MetricStub]],
    *,
    agent_x_values: list[float] | None = None,
    table_results: dict[_AlgorithmStub, dict[_MetricStub, dict[str, tuple[float, float]]]] | None = None,
    plot_results: dict[_AlgorithmStub, dict[_MetricStub, tuple[list[float], list[float], list[float], list[float]]]] | None = None,
) -> MetricResult:
    if agent_x_values is None:
        agent_x_values = [1.0] * len(algorithms)

    flat_plot_metrics = plot_metrics if isinstance(plot_metrics[0], _MetricStub) else [
        metric for group in plot_metrics for metric in group
    ]

    default_table_results: dict[_AlgorithmStub, dict[_MetricStub, dict[str, tuple[float, float]]]] = {}
    default_plot_results: dict[_AlgorithmStub, dict[_MetricStub, tuple[list[float], list[float], list[float], list[float]]]] = {}

    for alg_idx, alg in enumerate(algorithms):
        default_table_results[alg] = {}
        default_plot_results[alg] = {}
        for metric_idx, metric in enumerate(table_metrics):
            value = float(alg_idx + metric_idx + 1)
            default_table_results[alg][metric] = {"avg": (value, 0.0)}
        for metric_idx, metric in enumerate(flat_plot_metrics):
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
    metric = _MetricStub("table one", "plot one")

    metrics_result = _build_display_metric_result([alg_a, alg_b], [metric], [[metric]])
    algorithms = [alg_b] if algorithm_filter == "object" else ["A"]
    captured = _run_display_with_capture(monkeypatch, metrics_result, algorithms=algorithms)

    assert "table" in captured
    assert "plot" in captured
    assert [alg.name for alg in captured["table"].table_results] == expected_algorithms
    assert [alg.name for alg in captured["table"].plot_results] == expected_algorithms
    assert [alg.name for alg in captured["table"].agent_metrics] == expected_algorithms


def test_display_metrics_keeps_nan_table_metrics(monkeypatch) -> None:  # noqa: D103
    alg_a = _AlgorithmStub("A")
    alg_b = _AlgorithmStub("B")
    valid_metric = _MetricStub("valid table", "valid plot")
    nan_metric = _MetricStub("nan table", "nan plot")

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
    assert [metric.table_description for metric in captured["table"].table_metrics] == ["valid table", "nan table"]
    assert [metric.plot_description for metric in captured["table"].plot_metrics] == ["valid plot", "nan plot"]
    for algorithm_results in captured["table"].table_results.values():
        assert [metric.table_description for metric in algorithm_results] == ["valid table", "nan table"]


def test_display_metrics_filters_metrics_by_name(monkeypatch) -> None:  # noqa: D103
    alg_a = _AlgorithmStub("A")
    metric_1 = _MetricStub("table one", "plot one")
    metric_2 = _MetricStub("table two", "plot two")

    metrics_result = _build_display_metric_result([alg_a], [metric_1, metric_2], [[metric_1], [metric_2]])

    captured = _run_display_with_capture(
        monkeypatch,
        metrics_result,
        table_metrics=["table two"],
        plot_metrics=[["plot one"]],
    )

    assert "table" in captured
    assert [metric.table_description for metric in captured["table"].table_metrics] == ["table two"]
    assert isinstance(captured["table"].plot_metrics, list)
    grouped_plot_metrics = captured["table"].plot_metrics
    assert isinstance(grouped_plot_metrics[0], list)
    assert [metric.plot_description for metric in grouped_plot_metrics[0]] == ["plot one"]


def test_display_metrics_filters_metrics_with_mixed_objects_and_names(monkeypatch) -> None:  # noqa: D103
    alg_a = _AlgorithmStub("A")
    metric_1 = _MetricStub("table one", "plot one")
    metric_2 = _MetricStub("table two", "plot two")

    metrics_result = _build_display_metric_result([alg_a], [metric_1, metric_2], [[metric_1], [metric_2]])

    captured = _run_display_with_capture(
        monkeypatch,
        metrics_result,
        table_metrics=[metric_1, "table two"],
        plot_metrics=[[metric_1, "plot two"]],
    )

    assert "table" in captured
    assert [metric.table_description for metric in captured["table"].table_metrics] == ["table one", "table two"]
    assert isinstance(captured["table"].plot_metrics, list)
    grouped_plot_metrics = captured["table"].plot_metrics
    assert isinstance(grouped_plot_metrics[0], list)
    assert [metric.plot_description for metric in grouped_plot_metrics[0]] == ["plot one", "plot two"]


def test_display_metrics_rejects_mixed_shape_plot_metrics() -> None:  # noqa: D103
    alg_a = _AlgorithmStub("A")
    metric_1 = _MetricStub("table one", "plot one")
    metric_2 = _MetricStub("table two", "plot two")

    metrics_result = _build_display_metric_result([alg_a], [metric_1, metric_2], [[metric_1], [metric_2]])

    with pytest.raises(ValueError, match="all items must be lists"):
        display_metrics(
            metrics_result=metrics_result,
            plot_metrics=[metric_1, [metric_2]],
        )


def test_display_metrics_filters_algorithms_with_mixed_objects_and_names(monkeypatch) -> None:  # noqa: D103
    alg_a = _AlgorithmStub("A")
    alg_b = _AlgorithmStub("B")
    alg_c = _AlgorithmStub("C")
    metric = _MetricStub("table one", "plot one")

    metrics_result = _build_display_metric_result([alg_a, alg_b, alg_c], [metric], [[metric]])

    captured = _run_display_with_capture(monkeypatch, metrics_result, algorithms=[alg_a, "C"])

    assert "table" in captured
    assert [alg.name for alg in captured["table"].table_results] == ["A", "C"]
    assert [alg.name for alg in captured["table"].plot_results] == ["A", "C"]
    assert [alg.name for alg in captured["table"].agent_metrics] == ["A", "C"]


def test_display_metrics_raises_when_all_algorithms_filtered_out(monkeypatch) -> None:  # noqa: D103
    alg_a = _AlgorithmStub("A")
    metric = _MetricStub("table one", "plot one")

    metrics_result = MetricResult(
        agent_metrics={alg_a: [[_agent_metrics_view(1.0)]]},
        table_metrics=[metric],
        plot_metrics=[[metric]],
        table_results={alg_a: {metric: {"avg": (1.0, 0.0)}}},
        plot_results={alg_a: {metric: ([0.0], [1.0], [1.0], [1.0])}},
    )

    with pytest.raises(ValueError, match="No algorithms remain after filtering"):
        display_metrics(metrics_result=metrics_result, algorithms=["NonExistent"])


def test_display_metrics_raises_when_all_table_and_plot_metrics_filtered_out(monkeypatch) -> None:  # noqa: D103
    alg_a = _AlgorithmStub("A")
    metric_1 = _MetricStub("table one", "plot one")
    metric_2 = _MetricStub("table two", "plot two")

    metrics_result = MetricResult(
        agent_metrics={alg_a: [[_agent_metrics_view(1.0)]]},
        table_metrics=[metric_1, metric_2],
        plot_metrics=[[metric_1], [metric_2]],
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
    metric_1 = _MetricStub("table one", "plot one")
    metric_2 = _MetricStub("table two", "plot two")

    metrics_result = _build_display_metric_result([alg_a], [metric_1, metric_2], [[metric_1], [metric_2]])

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
    metric_1 = _MetricStub("table one", "plot one")
    metric_2 = _MetricStub("table two", "plot two")

    metrics_result = _build_display_metric_result([alg_a], [metric_1, metric_2], [[metric_1], [metric_2]])

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
    metric_1 = _MetricStub("table one", "plot one")
    metric_2 = _MetricStub("table two", "plot two")

    metrics_result = MetricResult(
        agent_metrics={alg_a: [[[]]], alg_b: [[[]]]},
        table_metrics=[metric_1, metric_1, metric_2],
        plot_metrics=[[metric_1], [metric_1, metric_2]],
        table_results={alg_a: {metric_1: {"avg": (1.0, 0.1)}}},
        plot_results={alg_b: {metric_2: ([0.0], [2.0], [2.0], [2.0])}},
    )

    assert metrics_result.available_algorithms == ["A", "B"]
    assert metrics_result.available_table_metrics == ["table one", "table two"]
    assert metrics_result.available_plot_metrics == ["plot one", "plot two"]


def test_compute_metrics_rejects_duplicate_table_metric_descriptions(monkeypatch) -> None:  # noqa: D103
    benchmark_result = _build_minimal_benchmark_result()
    metric_1 = _MetricStub("same table", "plot one")
    metric_2 = _MetricStub("same table", "plot two")

    monkeypatch.setattr("decent_bench.benchmark._compute.compute_tables", lambda *args, **kwargs: {})
    monkeypatch.setattr("decent_bench.benchmark._compute.compute_plots", lambda *args, **kwargs: {})

    with pytest.raises(ValueError, match="Table metric descriptions must be unique"):
        compute_metrics(benchmark_result=benchmark_result, table_metrics=[metric_1, metric_2], plot_metrics=[])


def test_compute_metrics_rejects_duplicate_plot_metric_descriptions(monkeypatch) -> None:  # noqa: D103
    benchmark_result = _build_minimal_benchmark_result()
    metric_1 = _MetricStub("table one", "same plot")
    metric_2 = _MetricStub("table two", "same plot")

    monkeypatch.setattr("decent_bench.benchmark._compute.compute_tables", lambda *args, **kwargs: {})
    monkeypatch.setattr("decent_bench.benchmark._compute.compute_plots", lambda *args, **kwargs: {})

    with pytest.raises(ValueError, match="Plot metric descriptions must be unique"):
        compute_metrics(benchmark_result=benchmark_result, table_metrics=[], plot_metrics=[[metric_1], [metric_2]])


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


def test_create_separate_legend_figure_tightly_fits_legend() -> None:  # noqa: D103
    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0], [0.0, 1.0], label="Algorithm A")
    ax.plot([0.0, 1.0], [1.0, 0.0], label="Algorithm B")
    handles, labels = ax.get_legend_handles_labels()

    legend_fig = _create_separate_legend_figure(handles, labels, label_cols=2, estimated_rows=1)

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

    plt.close(fig)
    plt.close(legend_fig)


def test_add_legend_and_save_creates_missing_parent_directory() -> None:  # noqa: D103
    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0], [0.0, 1.0], label="A")

    with tempfile.TemporaryDirectory(dir=".") as temp_dir:
        target = Path(temp_dir) / "nested" / "results" / "plot.png"
        assert not target.parent.exists()

        _add_legend_and_save(fig, [ax], plot_path=target)

        assert target.exists()


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


def test_is_available_default_returns_true() -> None:  # noqa: D103
    """Base Metric.is_available default: always available."""
    metric = _MetricStub("t", "p")
    available, reason = metric.is_available(SimpleNamespace())
    assert available
    assert reason is None


# -----------------------------------------------------------------------------
# Plot Metric Stub
# -----------------------------------------------------------------------------

class _PlotMetricStub(Metric):
    def __init__(self, plot_description: str, plot_data_by_x_value: dict[float, list[tuple[float, float]]]) -> None:
        super().__init__([np.average], fmt=".2e", y_log=False)
        self._plot_description = plot_description
        self._plot_data_by_x_value = plot_data_by_x_value

    @property
    def table_description(self) -> str:
        return self._plot_description

    @property
    def plot_description(self) -> str:
        return self._plot_description

    def get_data_from_trial(self, agents, problem, iteration):  # noqa: D102
        return [0.0]

    def get_plot_data(self, agents, problem):  # noqa: D102
        last_iteration = agents[0].x_history.max()
        x_value = float(agents[0].x_history[last_iteration][0])
        return self._plot_data_by_x_value[x_value]
