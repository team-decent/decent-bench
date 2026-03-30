import numpy as np
from types import SimpleNamespace

import decent_bench.metrics.metric_utils as metric_utils
from decent_bench.agents import AgentHistory, AgentMetricsView
from decent_bench.benchmark._display import display_metrics
from decent_bench.benchmark._metric_result import MetricResult
from decent_bench.metrics._metric import Metric
from decent_bench.metrics._plots import MAX_LOG_PLOT_VALUE, compute_plots
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


# -----------------------------------------------------------------------------
# display_metrics Filtering Tests
# -----------------------------------------------------------------------------

def test_display_metrics_filters_algorithms(monkeypatch) -> None:  # noqa: D103
    alg_a = _AlgorithmStub("A")
    alg_b = _AlgorithmStub("B")

    metrics_result = MetricResult(
        agent_metrics={alg_a: [[[]]], alg_b: [[[]]]},
        table_metrics=[],
        plot_metrics=[],
        table_results={alg_a: {}, alg_b: {}},
        plot_results={alg_a: {}, alg_b: {}},
    )

    captured = _run_display_with_capture(monkeypatch, metrics_result, algorithms=[alg_b])

    assert "table" in captured
    assert "plot" in captured
    assert [alg.name for alg in captured["table"].table_results] == ["B"]
    assert [alg.name for alg in captured["table"].plot_results] == ["B"]
    assert [alg.name for alg in captured["table"].agent_metrics] == ["B"]


def test_display_metrics_keeps_nan_table_metrics(monkeypatch) -> None:  # noqa: D103
    alg_a = _AlgorithmStub("A")
    alg_b = _AlgorithmStub("B")
    valid_metric = _MetricStub("valid table", "valid plot")
    nan_metric = _MetricStub("nan table", "nan plot")

    metrics_result = MetricResult(
        agent_metrics={alg_a: [[_agent_metrics_view(1.0)]], alg_b: [[_agent_metrics_view(2.0)]]},
        table_metrics=[valid_metric, nan_metric],
        plot_metrics=[valid_metric, nan_metric],
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
            1.0: [(0.0, 1.0), (1.0, 2.0), (2.0, MAX_LOG_PLOT_VALUE * 10)],
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
# Metric Availability Tests
# -----------------------------------------------------------------------------

def test_xerror_marks_unavailable_without_x_optimal() -> None:  # noqa: D103
    x_error = XError([np.average])
    x_error.clear_unavailable()
    _ = x_error.get_table_data([_agent_metrics_view(1.0)], SimpleNamespace(x_optimal=None))
    assert x_error.is_unavailable
    assert x_error.unavailable_reason == "requires problem.x_optimal"


def test_xerror_not_unavailable_with_x_optimal() -> None:  # noqa: D103
    x_error = XError([np.average])
    x_error.clear_unavailable()
    _ = x_error.get_table_data([_agent_metrics_view(1.0)], SimpleNamespace(x_optimal=np.array([0.0])))
    assert not x_error.is_unavailable


def test_metrics_short_circuit_without_test_data(monkeypatch) -> None:  # noqa: D103
    def _fail(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("metric util should not be called without test_data")

    monkeypatch.setattr(metric_utils, "accuracy", _fail)
    monkeypatch.setattr(metric_utils, "mse", _fail)
    monkeypatch.setattr(metric_utils, "precision", _fail)
    monkeypatch.setattr(metric_utils, "recall", _fail)

    agents = [_agent_metrics_view(1.0), _agent_metrics_view(2.0)]
    problem = SimpleNamespace(test_data=None)
    metrics = [Accuracy([np.average]), MSE([np.average]), Precision([np.average]), Recall([np.average])]

    for metric in metrics:
        metric.clear_unavailable()
        values = metric.get_table_data(agents, problem)
        assert metric.is_unavailable
        assert metric.unavailable_reason == "requires problem.test_data"
        assert len(values) == len(agents)
        assert all(np.isnan(value) for value in values)


def test_metrics_short_circuit_without_empirical_risk_cost(monkeypatch) -> None:  # noqa: D103
    def _fail(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("metric util should not be called without EmpiricalRiskCost")

    monkeypatch.setattr(metric_utils, "accuracy", _fail)
    monkeypatch.setattr(metric_utils, "mse", _fail)
    monkeypatch.setattr(metric_utils, "precision", _fail)
    monkeypatch.setattr(metric_utils, "recall", _fail)
    monkeypatch.setattr(metric_utils, "split_dataset", _fail)

    agents = [_agent_metrics_view(1.0), _agent_metrics_view(2.0)]
    # test_data exists, so any NaN output must come from non-EmpiricalRiskCost short-circuit.
    problem = SimpleNamespace(test_data=[(np.array([0.0]), 0)])
    metrics = [Accuracy([np.average]), MSE([np.average]), Precision([np.average]), Recall([np.average])]

    for metric in metrics:
        metric.clear_unavailable()
        values = metric.get_table_data(agents, problem)
        assert metric.is_unavailable
        assert metric.unavailable_reason == "requires EmpiricalRiskCost agents"
        assert len(values) == len(agents)
        assert all(np.isnan(value) for value in values)


def test_classification_metrics_short_circuit_without_integer_targets(monkeypatch) -> None:  # noqa: D103
    def _fail(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("classification util should not be called with non-integer targets")

    monkeypatch.setattr(metric_utils, "accuracy", _fail)
    monkeypatch.setattr(metric_utils, "precision", _fail)
    monkeypatch.setattr(metric_utils, "recall", _fail)
    monkeypatch.setattr(metric_utils, "split_dataset", lambda _: ((), np.array([0.1, 0.2])))

    # Patch the class used in isinstance checks so these stub agents pass the EmpiricalRiskCost gate.
    monkeypatch.setattr("decent_bench.metrics.metric_library.costs.EmpiricalRiskCost", object)

    agents = [_agent_metrics_view(1.0), _agent_metrics_view(2.0)]
    problem = SimpleNamespace(test_data=[(np.array([0.0]), 0.1)])
    metrics = [Accuracy([np.average]), Precision([np.average]), Recall([np.average])]

    for metric in metrics:
        metric.clear_unavailable()
        values = metric.get_table_data(agents, problem)
        assert metric.is_unavailable
        assert metric.unavailable_reason == "requires integer targets"
        assert len(values) == len(agents)
        assert all(np.isnan(value) for value in values)


def test_regret_marks_unavailable_without_x_optimal() -> None:  # noqa: D103
    regret = Regret([np.average])
    regret.clear_unavailable()
    _ = regret.get_table_data([_agent_metrics_view(1.0)], SimpleNamespace(x_optimal=None))
    assert regret.is_unavailable
    assert regret.unavailable_reason == "requires problem.x_optimal"


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
