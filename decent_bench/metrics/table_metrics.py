import pathlib
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
import tabulate as tb
from numpy import linalg as la
from scipy import stats

import decent_bench.metrics.metric_utils as utils
import decent_bench.utils.interoperability as iop
from decent_bench.agents import AgentMetricsView
from decent_bench.benchmark_problem import BenchmarkProblem
from decent_bench.distributed_algorithms import Algorithm
from decent_bench.utils.logger import LOGGER

Statistic = Callable[[Sequence[float]], float]


class TableMetric(ABC):
    """
    Metric to display in the statistical results table at the end of the benchmarking execution.

    Args:
        statistics: sequence of statistics such as :func:`min`, :func:`sum`, and :func:`~numpy.average` used for
            aggregating the data retrieved with :func:`get_data_from_trial` into a single value, each statistic gets its
            own row in the table
        fmt: format string used to format the values in the table, defaults to ".2e". Common formats include:
            - ".2e": scientific notation with 2 decimal places
            - ".3f": fixed-point notation with 3 decimal places
            - ".4g": general format with 4 significant digits
            - ".1%": percentage format with 1 decimal place

            Where the integer specifies the precision.
            See :meth:`str.format` documentation for details on the format string options.

    """

    def __init__(self, statistics: list[Statistic], fmt: str = ".2e"):
        self.statistics = statistics
        self.fmt = fmt

    @property
    def can_diverge(self) -> bool:
        """
        Indicates whether the metric can diverge, i.e. take on infinite or NaN values.

        If True then the table will try to indicate if the has metric diverged.
        """
        return True

    @property
    @abstractmethod
    def table_description(self) -> str:
        """Metric description to display in the table."""

    @abstractmethod
    def get_data_from_trial(self, agents: list[AgentMetricsView], problem: BenchmarkProblem) -> Sequence[float]:
        """Extract trial data to be aggregated into a single value by each of the *statistics*."""


class Regret(TableMetric):
    """
    Global regret using the agents' final x.

    Global regret is defined as:

    .. include:: snippets/global_cost_error.rst
    """

    table_description: str = "regret \n[<1e-9 = exact conv.]"

    def get_data_from_trial(self, agents: list[AgentMetricsView], problem: BenchmarkProblem) -> tuple[float]:  # noqa: D102
        return (utils.regret(agents, problem, iteration=-1),)


class GradientNorm(TableMetric):
    """
    Global gradient norm using the agents' final x.

    Global gradient norm is defined as:

    .. include:: snippets/global_gradient_optimality.rst
    """

    table_description: str = "gradient norm"

    def get_data_from_trial(self, agents: list[AgentMetricsView], _: BenchmarkProblem) -> tuple[float]:  # noqa: D102
        return (utils.gradient_norm(agents, iteration=-1),)


class XError(TableMetric):
    r"""
    X error per agent as defined below.

    .. math::
        \{ \|\mathbf{x}_i - \mathbf{x}^\star\|, \|\mathbf{x}_j - \mathbf{x}^\star\|, ... \}

    where :math:`\mathbf{x}_i` is agent i's final x,
    :math:`\mathbf{x}_j` is agent j's final x,
    and :math:`\mathbf{x}^\star` is the optimal x defined in the *problem*.

    """

    table_description: str = "x error"

    def get_data_from_trial(self, agents: list[AgentMetricsView], problem: BenchmarkProblem) -> list[float]:  # noqa: D102
        return [
            float(la.norm(iop.to_numpy(problem.x_optimal) - iop.to_numpy(a.x_history[max(a.x_history)])))
            for a in agents
        ]


class AsymptoticConvergenceOrder(TableMetric):
    """
    Asymptotic convergence order per agent as defined below.

    .. include:: snippets/asymptotic_convergence_rate_and_order.rst
    """

    table_description: str = "asymptotic convergence order"

    def get_data_from_trial(self, agents: list[AgentMetricsView], problem: BenchmarkProblem) -> list[float]:  # noqa: D102
        return [utils.asymptotic_convergence_rate_and_order(a, problem)[1] for a in agents]


class AsymptoticConvergenceRate(TableMetric):
    """
    Asymptotic convergence rate per agent as defined below.

    .. include:: snippets/asymptotic_convergence_rate_and_order.rst
    """

    table_description: str = "asymptotic convergence rate"

    def get_data_from_trial(self, agents: list[AgentMetricsView], problem: BenchmarkProblem) -> list[float]:  # noqa: D102
        return [utils.asymptotic_convergence_rate_and_order(a, problem)[0] for a in agents]


class IterativeConvergenceOrder(TableMetric):
    """
    Iterative convergence order per agent as defined below.

    .. include:: snippets/iterative_convergence_rate_and_order.rst
    """

    table_description: str = "iterative convergence order"

    def get_data_from_trial(self, agents: list[AgentMetricsView], problem: BenchmarkProblem) -> list[float]:  # noqa: D102
        return [utils.iterative_convergence_rate_and_order(a, problem)[1] for a in agents]


class IterativeConvergenceRate(TableMetric):
    """
    Iterative convergence rate per agent as defined below.

    .. include:: snippets/iterative_convergence_rate_and_order.rst
    """

    table_description: str = "iterative convergence rate"

    def get_data_from_trial(self, agents: list[AgentMetricsView], problem: BenchmarkProblem) -> list[float]:  # noqa: D102
        return [utils.iterative_convergence_rate_and_order(a, problem)[0] for a in agents]


class XUpdates(TableMetric):
    """Number of iterations/updates of x per agent."""

    table_description: str = "nr x updates"

    def get_data_from_trial(self, agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[float]:  # noqa: D102
        return [a.n_x_updates for a in agents]


class FunctionCalls(TableMetric):
    """Number of cost function evaluate calls per agent."""

    table_description: str = "nr function calls"

    def get_data_from_trial(self, agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[float]:  # noqa: D102
        return [a.n_function_calls for a in agents]


class GradientCalls(TableMetric):
    """Number of cost function gradient calls per agent."""

    table_description: str = "nr gradient calls"

    def get_data_from_trial(self, agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[float]:  # noqa: D102
        return [a.n_gradient_calls for a in agents]


class HessianCalls(TableMetric):
    """Number of cost function hessian calls per agent."""

    table_description: str = "nr hessian calls"

    def get_data_from_trial(self, agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[float]:  # noqa: D102
        return [a.n_hessian_calls for a in agents]


class ProximalCalls(TableMetric):
    """Number of cost function proximal calls per agent."""

    table_description: str = "nr proximal calls"

    def get_data_from_trial(self, agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[float]:  # noqa: D102
        return [a.n_proximal_calls for a in agents]


class SentMessages(TableMetric):
    """Number of sent messages per agent."""

    table_description: str = "nr sent messages"

    def get_data_from_trial(self, agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[float]:  # noqa: D102
        return [a.n_sent_messages for a in agents]


class ReceivedMessages(TableMetric):
    """Number of received messages per agent."""

    table_description: str = "nr received messages"

    def get_data_from_trial(self, agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[float]:  # noqa: D102
        return [a.n_received_messages for a in agents]


class SentMessagesDropped(TableMetric):
    """Number of sent messages that were dropped per agent."""

    table_description: str = "nr sent messages dropped"

    def get_data_from_trial(self, agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[float]:  # noqa: D102
        return [a.n_sent_messages_dropped for a in agents]


class Accuracy(TableMetric):
    """
    Final accuracy per agent.

    Only applicable for :class:`~decent_bench.costs.EmpiricalRiskCost`, returns NaN otherwise.

    See :func:`~decent_bench.metrics.metric_utils.accuracy` for the specific accuracy calculation used.
    """

    table_description: str = "accuracy"
    can_diverge: bool = False

    def get_data_from_trial(self, agents: list[AgentMetricsView], problem: BenchmarkProblem) -> list[float]:  # noqa: D102
        return utils.accuracy(agents, problem, iteration=-1)


class OptimalAccuracy(TableMetric):
    """
    Accuracy calculated using the optimal x defined in the benchmark problem instead of the agents' final x.

    Only applicable for :class:`~decent_bench.costs.EmpiricalRiskCost`, returns NaN otherwise.

    See :func:`~decent_bench.metrics.metric_utils.optimal_x_accuracy` for the specific optimal
    accuracy calculation used.
    """

    table_description: str = "optimal accuracy"
    can_diverge: bool = False

    def get_data_from_trial(self, _: list[AgentMetricsView], problem: BenchmarkProblem) -> tuple[float]:  # noqa: D102
        return (utils.optimal_x_accuracy(problem),)


class MSE(TableMetric):
    """
    Final MSE per agent, only applicable for :class:`~decent_bench.costs.EmpiricalRiskCost`.

    See :func:`~decent_bench.metrics.metric_utils.mse` for the specific MSE calculation used.
    """

    table_description: str = "mse"
    can_diverge: bool = False

    def get_data_from_trial(self, agents: list[AgentMetricsView], problem: BenchmarkProblem) -> list[float]:  # noqa: D102
        return utils.mse(agents, problem, iteration=-1)


class OptimalMSE(TableMetric):
    """
    MSE calculated using the optimal x defined in the benchmark problem instead of the agents' final x.

    Only applicable for :class:`~decent_bench.costs.EmpiricalRiskCost`, returns NaN otherwise.

    See :func:`~decent_bench.metrics.metric_utils.optimal_x_mse` for the specific optimal MSE calculation used.
    """

    table_description: str = "optimal mse"
    can_diverge: bool = False

    def get_data_from_trial(self, _: list[AgentMetricsView], problem: BenchmarkProblem) -> tuple[float]:  # noqa: D102
        return (utils.optimal_x_mse(problem),)


class Precision(TableMetric):
    """
    Final precision per agent.

    Only applicable for :class:`~decent_bench.costs.EmpiricalRiskCost`, returns NaN otherwise.

    See :func:`~decent_bench.metrics.metric_utils.precision` for the specific precision calculation used.
    """

    table_description: str = "precision"
    can_diverge: bool = False

    def get_data_from_trial(self, agents: list[AgentMetricsView], problem: BenchmarkProblem) -> list[float]:  # noqa: D102
        return utils.precision(agents, problem, iteration=-1)


class OptimalPrecision(TableMetric):
    """
    Precision calculated using the optimal x defined in the benchmark problem instead of the agents' final x.

    Only applicable for :class:`~decent_bench.costs.EmpiricalRiskCost`, returns NaN otherwise.

    See :func:`~decent_bench.metrics.metric_utils.optimal_x_precision` for the specific
    optimal precision calculation used.
    """

    table_description: str = "optimal precision"
    can_diverge: bool = False

    def get_data_from_trial(self, _: list[AgentMetricsView], problem: BenchmarkProblem) -> tuple[float]:  # noqa: D102
        return (utils.optimal_x_precision(problem),)


class Recall(TableMetric):
    """
    Final recall per agent.

    Only applicable for :class:`~decent_bench.costs.EmpiricalRiskCost`, returns NaN otherwise.

    See :func:`~decent_bench.metrics.metric_utils.recall` for the specific recall calculation used.
    """

    table_description: str = "recall"
    can_diverge: bool = False

    def get_data_from_trial(self, agents: list[AgentMetricsView], problem: BenchmarkProblem) -> list[float]:  # noqa: D102
        return utils.recall(agents, problem, iteration=-1)


class OptimalRecall(TableMetric):
    """
    Recall calculated using the optimal x defined in the benchmark problem instead of the agents' final x.

    Only applicable for :class:`~decent_bench.costs.EmpiricalRiskCost`, returns NaN otherwise.

    See :func:`~decent_bench.metrics.metric_utils.optimal_x_recall` for the specific optimal recall calculation used.
    """

    table_description: str = "optimal recall"
    can_diverge: bool = False

    def get_data_from_trial(self, _: list[AgentMetricsView], problem: BenchmarkProblem) -> tuple[float]:  # noqa: D102
        return (utils.optimal_x_recall(problem),)


DEFAULT_TABLE_METRICS: list[TableMetric] = [
    Regret([utils.single]),
    GradientNorm([utils.single]),
    XError([min, np.average, max]),
    AsymptoticConvergenceOrder([np.average]),
    AsymptoticConvergenceRate([np.average]),
    IterativeConvergenceOrder([np.average]),
    IterativeConvergenceRate([np.average]),
    XUpdates([np.average, sum]),
    FunctionCalls([np.average, sum]),
    GradientCalls([np.average, sum]),
    HessianCalls([np.average, sum]),
    ProximalCalls([np.average, sum]),
    SentMessages([np.average, sum]),
    ReceivedMessages([np.average, sum]),
    SentMessagesDropped([np.average, sum]),
]
"""
- :class:`Regret` - :func:`~.metric_utils.single`
- :class:`GradientNorm` - :func:`~.metric_utils.single`
- :class:`XError` - :func:`min`, :func:`~numpy.average`, :func:`max`
- :class:`AsymptoticConvergenceOrder` - :func:`~numpy.average`
- :class:`AsymptoticConvergenceRate` - :func:`~numpy.average`
- :class:`IterativeConvergenceOrder` - :func:`~numpy.average`
- :class:`IterativeConvergenceRate` - :func:`~numpy.average`
- :class:`XUpdates` - :func:`~numpy.average`, :func:`sum`
- :class:`FunctionCalls` - :func:`~numpy.average`, :func:`sum`
- :class:`GradientCalls` - :func:`~numpy.average`, :func:`sum`
- :class:`HessianCalls` - :func:`~numpy.average`, :func:`sum`
- :class:`ProximalCalls` - :func:`~numpy.average`, :func:`sum`
- :class:`SentMessages` - :func:`~numpy.average`, :func:`sum`
- :class:`ReceivedMessages` - :func:`~numpy.average`, :func:`sum`
- :class:`SentMessagesDropped` - :func:`~numpy.average`, :func:`sum`

:meta hide-value:
"""

EMPIRICAL_TABLE_METRICS: list[TableMetric] = [
    Accuracy([min, np.average, max], fmt=".2%"),
    OptimalAccuracy([utils.single], fmt=".2%"),
    MSE([min, np.average, max]),
    OptimalMSE([utils.single]),
    Precision([min, np.average, max], fmt=".2%"),
    OptimalPrecision([utils.single], fmt=".2%"),
    Recall([min, np.average, max], fmt=".2%"),
    OptimalRecall([utils.single], fmt=".2%"),
]
"""
- :class:`Accuracy` - :func:`min`, :func:`~numpy.average`, :func:`max` with percentage format
- :class:`OptimalAccuracy` - :func:`~.metric_utils.single` with percentage format
- :class:`MSE` - :func:`min`, :func:`~numpy.average`, :func:`max`
- :class:`OptimalMSE` - :func:`~.metric_utils.single`
- :class:`Precision` - :func:`min`, :func:`~numpy.average`, :func:`max` with percentage format
- :class:`OptimalPrecision` - :func:`~.metric_utils.single` with percentage
- :class:`Recall` - :func:`min`, :func:`~numpy.average`, :func:`max` with percentage format
- :class:`OptimalRecall` - :func:`~.metric_utils.single` with percentage format

:meta hide-value:
"""


TABLE_METRICS_DOC_LINK = "https://decent-bench.readthedocs.io/en/latest/api/decent_bench.metrics.table_metrics.html"


def tabulate(
    resulting_agent_states: dict[Algorithm, list[list[AgentMetricsView]]],
    problem: BenchmarkProblem,
    metrics: list[TableMetric],
    confidence_level: float,
    table_fmt: Literal["grid", "latex"],
    *,
    table_path: str | None = None,
) -> None:
    """
    Print table with confidence intervals, one row per metric and statistic, and one column per algorithm.

    Args:
        resulting_agent_states: resulting agent states from the trial executions, grouped by algorithm
        problem: benchmark problem whose properties, e.g.
            :attr:`~decent_bench.benchmark_problem.BenchmarkProblem.x_optimal`,
            are used for metric calculations
        metrics: metrics to calculate
        confidence_level: confidence level of the confidence intervals
        table_fmt: table format, grid is suitable for the terminal while latex can be copy-pasted into a latex document
        table_path: optional path to save the table as a text file, if not provided the table is not saved to a file

    """
    if not metrics:
        return
    LOGGER.info(f"Table metric definitions can be found here: {TABLE_METRICS_DOC_LINK}")
    algs = list(resulting_agent_states)
    headers = ["Metric (statistic)"] + [alg.name for alg in algs]
    rows: list[list[str]] = []
    statistics_abbr = {"average": "avg", "median": "mdn"}
    with warnings.catch_warnings(action="ignore"), utils.MetricProgressBar() as progress:
        n_statistics = sum(len(metric.statistics) for metric in metrics)
        table_task = progress.add_task("Generating table", total=n_statistics, status="")
        for metric in metrics:
            progress.update(table_task, status=f"Task: {metric.table_description}")
            data_per_trial = [_data_per_trial(resulting_agent_states[a], problem, metric) for a in algs]
            for statistic in metric.statistics:
                row = [f"{metric.table_description} ({statistics_abbr.get(statistic.__name__) or statistic.__name__})"]
                for i in range(len(algs)):
                    agg_data_per_trial = [statistic(trial) for trial in data_per_trial[i]]
                    mean, margin_of_error = _calculate_mean_and_margin_of_error(agg_data_per_trial, confidence_level)
                    formatted_confidence_interval = _format_confidence_interval(
                        mean,
                        margin_of_error,
                        metric.fmt,
                        metric.can_diverge,
                    )
                    row.append(formatted_confidence_interval)
                rows.append(row)
                progress.advance(table_task)
        progress.update(table_task, status="Finalizing table")
    formatted_table = tb.tabulate(rows, headers, tablefmt=table_fmt)
    LOGGER.info("\n" + formatted_table)
    if table_path:
        pathlib.Path(table_path).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(table_path).write_text(formatted_table, encoding="utf-8")


def _data_per_trial(
    agents_per_trial: list[list[AgentMetricsView]], problem: BenchmarkProblem, metric: TableMetric
) -> list[Sequence[float]]:
    data_per_trial: list[Sequence[float]] = []
    for agents in agents_per_trial:
        trial_data = metric.get_data_from_trial(agents, problem)
        data_per_trial.append(trial_data)

    return data_per_trial


def _calculate_mean_and_margin_of_error(data: list[float], confidence_level: float) -> tuple[float, float]:
    mean = np.mean(data)
    sem = stats.sem(data) if len(set(data)) > 1 else None
    raw_interval = (
        stats.t.interval(confidence=confidence_level, df=len(data) - 1, loc=mean, scale=sem) if sem else (mean, mean)
    )
    if np.isfinite(mean) and np.isfinite(raw_interval).all():
        return (float(mean), float(mean - raw_interval[0]))

    return np.nan, np.nan


def _format_confidence_interval(mean: float, margin_of_error: float, fmt: str, can_diverge: bool) -> str:
    if not _is_valid_float_format_spec(fmt):
        LOGGER.warning(f"Invalid format string '{fmt}', defaulting to scientific notation")
        fmt = ".2e"

    formatted_confidence_interval = f"{mean:{fmt}} \u00b1 {margin_of_error:{fmt}}"

    if any(np.isnan([mean, margin_of_error])) and can_diverge:
        formatted_confidence_interval += " (diverged?)"

    return formatted_confidence_interval


def _is_valid_float_format_spec(fmt: str) -> bool:
    """
    Validate that the given format spec can be used to format a float.

    This avoids attempting to format real values with an invalid format string.

    """
    try:
        f"{0.01:{fmt}}"
    except (ValueError, TypeError):
        return False
    return True
