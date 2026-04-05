"""Collection of pre-defined table and plot metrics."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

import decent_bench.metrics.metric_utils as utils
import decent_bench.utils.interoperability as iop
from decent_bench import costs
from decent_bench.agents import AgentMetricsView
from decent_bench.metrics._metric import Metric

if TYPE_CHECKING:
    from decent_bench.benchmark import BenchmarkProblem


class Regret(Metric):
    r"""
    Global regret.

    Table:
        Global regret using the agents' final x.

    Plot:
        Global regret (y-axis) per iteration (x-axis).

    Global regret is defined as:

    .. include:: snippets/global_cost_error.rst

    Note:
        Available only when ``problem.x_optimal`` is provided.

    """

    table_description: str = "regret"
    plot_description: str = "regret"

    def is_available(  # noqa: D102
        self,
        problem: "BenchmarkProblem",
    ) -> tuple[bool, str | None]:
        if getattr(problem, "x_optimal", None) is None:
            return False, "requires problem.x_optimal"
        return True, None

    def get_data_from_trial(  # noqa: D102
        self,
        agents: Sequence[AgentMetricsView],
        problem: "BenchmarkProblem",
        iteration: int,
    ) -> tuple[float]:
        return (utils._regret(agents, problem, iteration),)  # noqa: SLF001


class GradientNorm(Metric):
    r"""
    Global gradient norm.

    Table:
        Gradient norm using the agents' final x.

    Plot:
        Gradient norm (y-axis) per iteration (x-axis).

    Gradient norm is defined as:

    .. include:: snippets/global_gradient_optimality.rst

    """

    table_description: str = "gradient norm"
    plot_description: str = "gradient norm"

    def get_data_from_trial(  # noqa: D102
        self,
        agents: Sequence[AgentMetricsView],
        _: "BenchmarkProblem",
        iteration: int,
    ) -> tuple[float]:
        return (utils._gradient_norm(agents, iteration),)  # noqa: SLF001


class XError(Metric):
    r"""
    Distance to optimal solution.

    Table:
        Distance to optimal solution using the agents' final x.

    Plot:
        Distance to optimal solution (y-axis) per iteration (x-axis).

    X error per agent is defined as:

    .. math::
        \{ \|\mathbf{x}_i - \mathbf{x}^\star\|, \|\mathbf{x}_j - \mathbf{x}^\star\|, ... \}

    where :math:`\mathbf{x}_i` is agent i's final x,
    :math:`\mathbf{x}_j` is agent j's final x,
    and :math:`\mathbf{x}^\star` is the optimal x defined in the *problem*.

    Note:
        Available only when ``problem.x_optimal`` is provided.

    """

    table_description: str = "x error"
    plot_description: str = "x error"

    def is_available(  # noqa: D102
        self,
        problem: "BenchmarkProblem",
    ) -> tuple[bool, str | None]:
        if getattr(problem, "x_optimal", None) is None:
            return False, "requires problem.x_optimal"
        return True, None

    def get_data_from_trial(  # noqa: D102
        self,
        agents: Sequence[AgentMetricsView],
        problem: "BenchmarkProblem",
        iteration: int,
    ) -> list[float]:
        return [utils._x_error(a, problem, iteration) for a in agents]  # noqa: SLF001


class ConsensusError(Metric):
    r"""
    Distance to consensus.

    Table:
        Distance of the agents states from their current average.

    Plot:
        Distance to consensus (y-axis) per iteration (x-axis).

    The consensus error per agent is defined as:

    .. math::
        \{ \|\mathbf{x}_i - \bar{\mathbf{x}}\|, \|\mathbf{x}_j - \bar{\mathbf{x}}\|, ... \}

    where :math:`\mathbf{x}_i` is agent i's current state,
    and :math:`\bar{\mathbf{x}}` is the average of all agents' states.

    .. seealso::
        :class:`~decent_bench.metrics.runtime_library.RuntimeConsensusError` for the runtime version.

    """

    table_description: str = "consensus error"
    plot_description: str = "consensus error"

    def get_data_from_trial(  # noqa: D102
        self,
        agents: Sequence[AgentMetricsView],
        _: "BenchmarkProblem",
        iteration: int,
    ) -> list[float]:

        x_mean = utils.x_mean(tuple(agents), iteration)

        if iteration == -1:
            return [float(iop.norm(x_mean - a.x_history[a.x_history.max()])) for a in agents]
        return [float(iop.norm(x_mean - a.x_history[iteration])) for a in agents]


class XUpdates(Metric):
    r"""
    Number of x iterations/updates.

    Table:
        Number of x iterations/updates per agent.

    Plot:
        Number of x iterations/updates (y-axis) per iteration (x-axis).
        Will be a flat line as the number of x iterations/updates is only calculated at the end of the trial,
        not per iteration.

    """

    table_description: str = "nr x updates"
    plot_description: str = "nr x updates"

    def get_data_from_trial(  # noqa: D102
        self,
        agents: Sequence[AgentMetricsView],
        _: "BenchmarkProblem",
        __: int,
    ) -> list[int]:
        return [a.n_x_updates for a in agents]


class FunctionCalls(Metric):
    r"""
    Number of function calls.

    Table:
        Number of function calls per agent.

    Plot:
        Number of function calls (y-axis) per iteration (x-axis).
        Will be a flat line as the number of function calls is only calculated at the end of the trial,
        not per iteration.

    Note:
        Can be a floating point number if :class:`~decent_bench.costs.EmpiricalRiskCost` is used and a
        batch size other than the full dataset size is used.

    """

    table_description: str = "nr function calls"
    plot_description: str = "nr function calls"

    def get_data_from_trial(  # noqa: D102
        self,
        agents: Sequence[AgentMetricsView],
        _: "BenchmarkProblem",
        __: int,
    ) -> list[float]:
        return [a.n_function_calls for a in agents]


class GradientCalls(Metric):
    r"""
    Number of gradient calls.

    Table:
        Number of gradient calls per agent.

    Plot:
        Number of gradient calls (y-axis) per iteration (x-axis).
        Will be a flat line as the number of gradient calls is only calculated at the end of the trial,
        not per iteration.

    Note:
        Can be a floating point number if :class:`~decent_bench.costs.EmpiricalRiskCost` is used and a
        batch size other than the full dataset size is used.

    """

    table_description: str = "nr gradient calls"
    plot_description: str = "nr gradient calls"

    def get_data_from_trial(  # noqa: D102
        self,
        agents: Sequence[AgentMetricsView],
        _: "BenchmarkProblem",
        __: int,
    ) -> list[float]:
        return [a.n_gradient_calls for a in agents]


class HessianCalls(Metric):
    r"""
    Number of Hessian calls.

    Table:
        Number of Hessian calls per agent.

    Plot:
        Number of Hessian calls (y-axis) per iteration (x-axis).
        Will be a flat line as the number of Hessian calls is only calculated at the end of the trial,
        not per iteration.

    Note:
        Can be a floating point number if :class:`~decent_bench.costs.EmpiricalRiskCost` is used and a
        batch size other than the full dataset size is used.

    """

    table_description: str = "nr Hessian calls"
    plot_description: str = "nr Hessian calls"

    def get_data_from_trial(  # noqa: D102
        self,
        agents: Sequence[AgentMetricsView],
        _: "BenchmarkProblem",
        __: int,
    ) -> list[float]:
        return [a.n_hessian_calls for a in agents]


class ProximalCalls(Metric):
    r"""
    Number of proximal calls.

    Table:
        Number of proximal calls per agent.

    Plot:
        Number of proximal calls (y-axis) per iteration (x-axis).
        Will be a flat line as the number of proximal calls is only calculated at the end of the trial,
        not per iteration.

    """

    table_description: str = "nr proximal calls"
    plot_description: str = "nr proximal calls"

    def get_data_from_trial(  # noqa: D102
        self,
        agents: Sequence[AgentMetricsView],
        _: "BenchmarkProblem",
        __: int,
    ) -> list[float]:
        return [a.n_proximal_calls for a in agents]


class SentMessages(Metric):
    r"""
    Number of sent messages.

    Table:
        Number of sent messages per agent.

    Plot:
        Number of sent messages (y-axis) per iteration (x-axis).
        Will be a flat line as the number of sent messages is calculated at the end of the trial,
        not per iteration.

    """

    table_description: str = "nr sent messages"
    plot_description: str = "nr sent messages"

    def get_data_from_trial(  # noqa: D102
        self,
        agents: Sequence[AgentMetricsView],
        _: "BenchmarkProblem",
        __: int,
    ) -> list[int]:
        return [a.n_sent_messages for a in agents]


class ReceivedMessages(Metric):
    r"""
    Number of received messages.

    Table:
        Number of received messages per agent.

    Plot:
        Number of received messages (y-axis) per iteration (x-axis).
        Will be a flat line as the number of received messages are calculated at the end of the trial,
        not per iteration.

    """

    table_description: str = "nr received messages"
    plot_description: str = "nr received messages"

    def get_data_from_trial(  # noqa: D102
        self,
        agents: Sequence[AgentMetricsView],
        _: "BenchmarkProblem",
        __: int,
    ) -> list[int]:
        return [a.n_received_messages for a in agents]


class SentMessagesDropped(Metric):
    r"""
    Number of sent messages dropped.

    Table:
        Number of sent messages dropped per agent.

    Plot:
        Number of sent messages dropped (y-axis) per iteration (x-axis).
        Will be a flat line as the number of sent messages dropped is calculated at the end of the trial,
        not per iteration.

    """

    table_description: str = "nr sent messages dropped"
    plot_description: str = "nr sent messages dropped"

    def get_data_from_trial(  # noqa: D102
        self,
        agents: Sequence[AgentMetricsView],
        _: "BenchmarkProblem",
        __: int,
    ) -> list[int]:
        return [a.n_sent_messages_dropped for a in agents]


class Accuracy(Metric):
    r"""
    Accuracy of the agents' predictions.

    Table:
        Accuracy of the agents' final x.

    Plot:
        Accuracy (y-axis) per iteration (x-axis).

        Accuracy is calculated as the mean accuracy across agents, where each agent's accuracy is calculated using its
        recorded x at that iteration.

    Only available for :class:`~decent_bench.costs.EmpiricalRiskCost` and integer targets.

    Accuracy measures the proportion of correct predictions:

    .. math::

        \text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}

    where TP, TN, FP, and FN are true positives, true negatives, false positives, and false negatives, respectively.

    Note:
        Available only when:

        - ``problem.test_data`` is provided,
        - all agent costs are :class:`~decent_bench.costs.EmpiricalRiskCost`,
        - target labels are integer-valued.

    """

    table_description: str = "accuracy"
    plot_description: str = "accuracy"
    can_diverge: bool = False

    def is_available(  # noqa: D102
        self,
        problem: "BenchmarkProblem",
    ) -> tuple[bool, str | None]:
        if getattr(problem, "test_data", None) is None:
            return False, "requires problem.test_data"
        if not all(isinstance(a.cost, costs.EmpiricalRiskCost) for a in problem.network.agents()):
            return False, "accuracy only applies if all agents have EmpiricalRiskCost"
        _, test_y = utils._split_dataset(problem.test_data)  # type: ignore[arg-type] # noqa: SLF001
        if test_y.dtype.kind not in {"i", "u"}:
            return False, f"accuracy only applies for integer targets, dtype {test_y.dtype} found"
        return True, None

    def get_data_from_trial(  # noqa: D102
        self,
        agents: Sequence[AgentMetricsView],
        problem: "BenchmarkProblem",
        iteration: int,
    ) -> list[float]:
        return utils._accuracy(agents, problem, iteration)  # noqa: SLF001


class MSE(Metric):
    r"""
    Mean squared error of the agents' predictions.

    Table:
        Mean squared error of the agents' final x.

    Plot:
        Mean Squared Error (MSE) (y-axis) per iteration (x-axis).

        MSE is calculated as the mean MSE across agents, where each agent's MSE is calculated using its
        recorded x at that iteration.

    Only available for :class:`~decent_bench.costs.EmpiricalRiskCost`.

    MSE measures the average squared difference between predictions and true values:

    .. math::

        \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2

    where :math:`\hat{y}_i` are the predicted values, :math:`y_i` are the true values, and :math:`n` is
    the number of samples.

    Note:
        Available only when ``problem.test_data`` is provided and all agent costs are
        :class:`~decent_bench.costs.EmpiricalRiskCost`.

    """

    table_description: str = "mse"
    plot_description: str = "mse"
    can_diverge: bool = False

    def is_available(  # noqa: D102
        self,
        problem: "BenchmarkProblem",
    ) -> tuple[bool, str | None]:
        if getattr(problem, "test_data", None) is None:
            return False, "requires problem.test_data"
        if not all(isinstance(a.cost, costs.EmpiricalRiskCost) for a in problem.network.agents()):
            return False, "MSE only applies if all agents have EmpiricalRiskCost"
        return True, None

    def get_data_from_trial(  # noqa: D102
        self,
        agents: Sequence[AgentMetricsView],
        problem: "BenchmarkProblem",
        iteration: int,
    ) -> list[float]:
        return utils._mse(agents, problem, iteration)  # noqa: SLF001


class Precision(Metric):
    r"""
    Precision of the agents' predictions.

    Table:
        Precision of the agents' final x.

    Plot:
        Precision (y-axis) per iteration (x-axis).

        Precision is calculated as the mean precision across agents, where each agent's precision is calculated using
        its recorded x at that iteration.

    Only available for :class:`~decent_bench.costs.EmpiricalRiskCost` and integer targets.

    Precision measures the proportion of positive predictions that are correct:

    .. math::

        \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}

    where TP is the number of true positives and FP is the number of false positives.

    Note:
        Available only when:

        - ``problem.test_data`` is provided,
        - all agent costs are :class:`~decent_bench.costs.EmpiricalRiskCost`,
        - target labels are integer-valued.

    """

    table_description: str = "precision"
    plot_description: str = "precision"
    can_diverge: bool = False

    def is_available(  # noqa: D102
        self,
        problem: "BenchmarkProblem",
    ) -> tuple[bool, str | None]:
        if getattr(problem, "test_data", None) is None:
            return False, "requires problem.test_data"
        if not all(isinstance(a.cost, costs.EmpiricalRiskCost) for a in problem.network.agents()):
            return False, "precision only applies if all agents have EmpiricalRiskCost"
        _, test_y = utils._split_dataset(problem.test_data)  # type: ignore[arg-type] # noqa: SLF001
        if test_y.dtype.kind not in {"i", "u"}:
            return False, f"precision only applies for integer targets, dtype {test_y.dtype} found"
        return True, None

    def get_data_from_trial(  # noqa: D102
        self,
        agents: Sequence[AgentMetricsView],
        problem: "BenchmarkProblem",
        iteration: int,
    ) -> list[float]:
        return utils._precision(agents, problem, iteration)  # noqa: SLF001


class Recall(Metric):
    r"""
    Recall of the agents' predictions.

    Table:
        Recall of the agents' final x.

    Plot:
        Recall (y-axis) per iteration (x-axis).

        Recall is calculated as the mean recall across agents, where each agent's recall is calculated using its
        recorded x at that iteration.

    Only available for :class:`~decent_bench.costs.EmpiricalRiskCost` and integer targets.

    Recall measures the proportion of actual positives that are correctly identified:

    .. math::

        \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}

    where TP is the number of true positives and FN is the number of false negatives.

    Note:
        Available only when:

        - ``problem.test_data`` is provided,
        - all agent costs are :class:`~decent_bench.costs.EmpiricalRiskCost`,
        - target labels are integer-valued.

    """

    table_description: str = "recall"
    plot_description: str = "recall"
    can_diverge: bool = False

    def is_available(  # noqa: D102
        self,
        problem: "BenchmarkProblem",
    ) -> tuple[bool, str | None]:
        if getattr(problem, "test_data", None) is None:
            return False, "requires problem.test_data"
        if not all(isinstance(a.cost, costs.EmpiricalRiskCost) for a in problem.network.agents()):
            return False, "recall only applies if all agents have EmpiricalRiskCost"
        _, test_y = utils._split_dataset(problem.test_data)  # type: ignore[arg-type]  # noqa: SLF001
        if test_y.dtype.kind not in {"i", "u"}:
            return False, f"recall only applies for integer targets, dtype {test_y.dtype} found"
        return True, None

    def get_data_from_trial(  # noqa: D102
        self,
        agents: Sequence[AgentMetricsView],
        problem: "BenchmarkProblem",
        iteration: int,
    ) -> list[float]:
        return utils._recall(agents, problem, iteration)  # noqa: SLF001


DEFAULT_TABLE_METRICS: list[Metric] = [
    Regret([utils.single]),
    GradientNorm([utils.single]),
    XError([min, np.average, max]),
    ConsensusError([min, np.average, max]),
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
- :class:`ConsensusError` - :func:`min`, :func:`~numpy.average`, :func:`max`
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

REGRESSION_TABLE_METRICS: list[Metric] = [
    MSE([min, np.average, max], x_log=False, y_log=True),
]
"""
- :class:`MSE` - :func:`min`, :func:`~numpy.average`, :func:`max`

:meta hide-value:
"""

CLASSIFICATION_TABLE_METRICS: list[Metric] = [
    Accuracy([min, np.average, max], fmt=".2%", x_log=False, y_log=False),
    Precision([min, np.average, max], fmt=".2%", x_log=False, y_log=False),
    Recall([min, np.average, max], fmt=".2%", x_log=False, y_log=False),
]
"""
- :class:`Accuracy` - :func:`min`, :func:`~numpy.average`, :func:`max` with percentage format
- :class:`Precision` - :func:`min`, :func:`~numpy.average`, :func:`max` with percentage format
- :class:`Recall` - :func:`min`, :func:`~numpy.average`, :func:`max` with percentage format

:meta hide-value:
"""

# No need to specify statistics for plot metrics as they are only
# used for table metrics, if you were to use the same Metric object
# for both, you would need to specify statistics
DEFAULT_PLOT_METRICS: list[Metric] = [
    Regret([], x_log=False, y_log=True),
    GradientNorm([], x_log=False, y_log=True),
    ConsensusError([], x_log=False, y_log=True),
]
"""
- :class:`Regret` (semi-log)
- :class:`GradientNorm` (semi-log)
- :class:`ConsensusError` (semi-log)

:meta hide-value:
"""


REGRESSION_PLOT_METRICS: list[Metric] = REGRESSION_TABLE_METRICS
"""
- :class:`MSE` (semi-log)

:meta hide-value:
"""

CLASSIFICATION_PLOT_METRICS: list[Metric] = CLASSIFICATION_TABLE_METRICS
"""
- :class:`Accuracy` (linear)
- :class:`Precision` (linear)
- :class:`Recall` (linear)

:meta hide-value:
"""
