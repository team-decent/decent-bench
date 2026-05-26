"""Collection of pre-defined table and plot metrics."""

from typing import TYPE_CHECKING

import numpy as np

import decent_bench.metrics.metric_utils as utils
import decent_bench.utils.interoperability as iop
from decent_bench.costs import Cost, EmpiricalRiskCost
from decent_bench.metrics._metric import Metric
from decent_bench.metrics._metrics_view import NetworkMetricsView
from decent_bench.networks import FedNetwork

if TYPE_CHECKING:
    from decent_bench.benchmark import BenchmarkProblem


class Regret(Metric):
    r"""
    Global regret.

    Table:
        Global regret using the agents'/clients' final x.

    Plot:
        Global regret (y-axis) per iteration (x-axis).

    Global regret is defined as:

    .. include:: snippets/global_cost_error.rst

    Note:
        Available only when ``problem.x_optimal`` is provided.

    """

    description: str = "regret"

    def is_available(  # noqa: D102
        self,
        problem: "BenchmarkProblem",
    ) -> tuple[bool, str | None]:
        if getattr(problem, "x_optimal", None) is None:
            return False, "requires problem.x_optimal"
        return True, None

    def get_data_from_trial(  # noqa: D102
        self,
        network: NetworkMetricsView,
        problem: "BenchmarkProblem",
        iteration: int,
    ) -> tuple[float]:
        return (utils._regret(network.agents(), problem, iteration),)  # noqa: SLF001


class GradientNorm(Metric):
    r"""
    Global gradient norm.

    Table:
        Gradient norm using the agents'/clients' final x.

    Plot:
        Gradient norm (y-axis) per iteration (x-axis).

    Gradient norm is defined as:

    .. include:: snippets/global_gradient_optimality.rst

    """

    description: str = "gradient norm"

    def get_data_from_trial(  # noqa: D102
        self,
        network: NetworkMetricsView,
        _: "BenchmarkProblem",
        iteration: int,
    ) -> tuple[float]:
        return (utils._gradient_norm(network.agents(), iteration),)  # noqa: SLF001


class XError(Metric):
    r"""
    Distance to optimal solution.

    Table:
        Distance to optimal solution using the mean of the agents'/clients' final x.

    Plot:
        Distance to optimal solution (y-axis) per iteration (x-axis).

    X error is defined as:

    .. math::
        \|\mathbf{\bar{x}} - \mathbf{x}^\star\|

    where  :math:`\mathbf{\bar{x}}` is the mean x across all agents/clients, and
    :math:`\mathbf{x}^\star` is the optimal x defined in the *problem*.


    Note:
        Available only when ``problem.x_optimal`` is provided.

    """

    description: str = "x error"

    def is_available(  # noqa: D102
        self,
        problem: "BenchmarkProblem",
    ) -> tuple[bool, str | None]:
        if getattr(problem, "x_optimal", None) is None:
            return False, "requires problem.x_optimal"
        return True, None

    def get_data_from_trial(  # noqa: D102
        self,
        network: NetworkMetricsView,
        problem: "BenchmarkProblem",
        iteration: int,
    ) -> tuple[float]:
        return (utils._x_error(network.agents(), problem, iteration),)  # noqa: SLF001


class ConsensusError(Metric):
    r"""
    Distance to consensus.

    Table:
        Distance of the agents'/clients' states from their current average.

    Plot:
        Distance to consensus (y-axis) per iteration (x-axis).

    The consensus error per agent/client is defined as:

    .. math::
        \{ \|\mathbf{x}_i - \bar{\mathbf{x}}\|, \|\mathbf{x}_j - \bar{\mathbf{x}}\|, ... \}

    where :math:`\mathbf{x}_i` is agent/client i's current state,
    :math:`\bar{\mathbf{x}}` is the average of all agents'/clients' states, and :math:`\| \cdot \|` is the 2-norm.

    .. seealso::
        :class:`~decent_bench.metrics.runtime_library.RuntimeConsensusError` for the runtime version.

    """

    description: str = "consensus error"

    def get_data_from_trial(  # noqa: D102
        self,
        network: NetworkMetricsView,
        _: "BenchmarkProblem",
        iteration: int,
    ) -> list[float]:
        agent_views = network.agents()
        x_mean = utils.x_mean(tuple(agent_views), iteration)
        return [float(iop.norm(x_mean - a.x_history[iteration])) for a in agent_views]


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

    description: str = "nr x updates"

    def get_data_from_trial(  # noqa: D102
        self,
        network: NetworkMetricsView,
        _: "BenchmarkProblem",
        __: int,
    ) -> list[int]:
        return [a.n_x_updates for a in network.agents()]


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

    description: str = "nr function calls"

    def get_data_from_trial(  # noqa: D102
        self,
        network: NetworkMetricsView,
        _: "BenchmarkProblem",
        __: int,
    ) -> list[float]:
        return [a.n_function_calls for a in network.agents()]


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

    description: str = "nr gradient calls"

    def get_data_from_trial(  # noqa: D102
        self,
        network: NetworkMetricsView,
        _: "BenchmarkProblem",
        __: int,
    ) -> list[float]:
        return [a.n_gradient_calls for a in network.agents()]


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

    description: str = "nr Hessian calls"

    def get_data_from_trial(  # noqa: D102
        self,
        network: NetworkMetricsView,
        _: "BenchmarkProblem",
        __: int,
    ) -> list[float]:
        return [a.n_hessian_calls for a in network.agents()]


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

    description: str = "nr proximal calls"

    def get_data_from_trial(  # noqa: D102
        self,
        network: NetworkMetricsView,
        _: "BenchmarkProblem",
        __: int,
    ) -> list[float]:
        return [a.n_proximal_calls for a in network.agents()]


class SentMessages(Metric):
    r"""
    Number of sent messages.

    Table:
        Number of sent messages per agent. For federated networks, this includes the server.

    Plot:
        Number of sent messages (y-axis) per iteration (x-axis).
        Will be a flat line as the number of sent messages is calculated at the end of the trial,
        not per iteration.

    """

    description: str = "nr sent messages"

    def get_data_from_trial(  # noqa: D102
        self,
        network: NetworkMetricsView,
        _: "BenchmarkProblem",
        __: int,
    ) -> list[float]:
        return [a.n_sent_messages for a in network.graph.nodes]


class ReceivedMessages(Metric):
    r"""
    Number of received messages.

    Table:
        Number of received messages per agent. For federated networks, this includes the server.

    Plot:
        Number of received messages (y-axis) per iteration (x-axis).
        Will be a flat line as the number of received messages are calculated at the end of the trial,
        not per iteration.

    """

    description: str = "nr received messages"

    def get_data_from_trial(  # noqa: D102
        self,
        network: NetworkMetricsView,
        _: "BenchmarkProblem",
        __: int,
    ) -> list[float]:
        return [a.n_received_messages for a in network.graph.nodes]


class SentMessagesDropped(Metric):
    r"""
    Number of sent messages dropped.

    Table:
        Number of sent messages dropped per agent. For federated networks, this includes the server.

    Plot:
        Number of sent messages dropped (y-axis) per iteration (x-axis).
        Will be a flat line as the number of sent messages dropped is calculated at the end of the trial,
        not per iteration.

    """

    description: str = "nr sent messages dropped"

    def get_data_from_trial(  # noqa: D102
        self,
        network: NetworkMetricsView,
        _: "BenchmarkProblem",
        __: int,
    ) -> list[float]:
        return [a.n_sent_messages_dropped for a in network.graph.nodes]


class Accuracy(Metric):
    r"""
    Accuracy of the agents'/clients' predictions.

    Table:
        Accuracy of the agents'/clients' final x.

    Plot:
        Accuracy (y-axis) per iteration (x-axis).

        Accuracy is calculated as the mean accuracy across agents/clients, where each agent's/client's accuracy is
        calculated using its recorded x at that iteration.

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

    description: str = "accuracy"

    def is_available(  # noqa: D102
        self,
        problem: "BenchmarkProblem",
    ) -> tuple[bool, str | None]:
        if getattr(problem, "test_data", None) is None:
            return False, "requires problem.test_data"
        if not all(isinstance(a.cost, EmpiricalRiskCost) for a in problem.network.agents()):
            return False, "accuracy only applies if all agents have EmpiricalRiskCost"
        _, test_y = utils._split_dataset(problem.test_data)  # type: ignore[arg-type] # noqa: SLF001
        if test_y.dtype.kind not in {"i", "u"}:
            return False, f"accuracy only applies for integer targets, dtype {test_y.dtype} found"
        return True, None

    def get_data_from_trial(  # noqa: D102
        self,
        network: NetworkMetricsView,
        problem: "BenchmarkProblem",
        iteration: int,
    ) -> list[float]:
        return utils._accuracy(network.agents(), problem, iteration)  # noqa: SLF001


class MSE(Metric):
    r"""
    Mean squared error of the agents'/clients' predictions.

    Table:
        Mean squared error of the agents'/clients' final x.

    Plot:
        Mean Squared Error (MSE) (y-axis) per iteration (x-axis).

        MSE is calculated as the mean MSE across agents/clients, where each agent's/client's MSE is calculated using
        its recorded x at that iteration.

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

    description: str = "mse"

    def is_available(  # noqa: D102
        self,
        problem: "BenchmarkProblem",
    ) -> tuple[bool, str | None]:
        if getattr(problem, "test_data", None) is None:
            return False, "requires problem.test_data"
        if not all(isinstance(a.cost, EmpiricalRiskCost) for a in problem.network.agents()):
            return False, "MSE only applies if all agents have EmpiricalRiskCost"
        return True, None

    def get_data_from_trial(  # noqa: D102
        self,
        network: NetworkMetricsView,
        problem: "BenchmarkProblem",
        iteration: int,
    ) -> list[float]:
        return utils._mse(network.agents(), problem, iteration)  # noqa: SLF001


class Precision(Metric):
    r"""
    Precision of the agents'/clients' predictions.

    Table:
        Precision of the agents'/clients' final x.

    Plot:
        Precision (y-axis) per iteration (x-axis).

        Precision is calculated as the mean precision across agents/clients, where each agent's/client's precision is
        calculated using its recorded x at that iteration.

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

    description: str = "precision"

    def is_available(  # noqa: D102
        self,
        problem: "BenchmarkProblem",
    ) -> tuple[bool, str | None]:
        if getattr(problem, "test_data", None) is None:
            return False, "requires problem.test_data"
        if not all(isinstance(a.cost, EmpiricalRiskCost) for a in problem.network.agents()):
            return False, "precision only applies if all agents have EmpiricalRiskCost"
        _, test_y = utils._split_dataset(problem.test_data)  # type: ignore[arg-type] # noqa: SLF001
        if test_y.dtype.kind not in {"i", "u"}:
            return False, f"precision only applies for integer targets, dtype {test_y.dtype} found"
        return True, None

    def get_data_from_trial(  # noqa: D102
        self,
        network: NetworkMetricsView,
        problem: "BenchmarkProblem",
        iteration: int,
    ) -> list[float]:
        return utils._precision(network.agents(), problem, iteration)  # noqa: SLF001


class Recall(Metric):
    r"""
    Recall of the agents'/clients' predictions.

    Table:
        Recall of the agents'/clients' final x.

    Plot:
        Recall (y-axis) per iteration (x-axis).

        Recall is calculated as the mean recall across agents/clients, where each agent's/client's recall is calculated
        using its recorded x at that iteration.

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

    description: str = "recall"

    def is_available(  # noqa: D102
        self,
        problem: "BenchmarkProblem",
    ) -> tuple[bool, str | None]:
        if getattr(problem, "test_data", None) is None:
            return False, "requires problem.test_data"
        if not all(isinstance(a.cost, EmpiricalRiskCost) for a in problem.network.agents()):
            return False, "recall only applies if all agents have EmpiricalRiskCost"
        _, test_y = utils._split_dataset(problem.test_data)  # type: ignore[arg-type]  # noqa: SLF001
        if test_y.dtype.kind not in {"i", "u"}:
            return False, f"recall only applies for integer targets, dtype {test_y.dtype} found"
        return True, None

    def get_data_from_trial(  # noqa: D102
        self,
        network: NetworkMetricsView,
        problem: "BenchmarkProblem",
        iteration: int,
    ) -> list[float]:
        return utils._recall(network.agents(), problem, iteration)  # noqa: SLF001


class Loss(Metric):
    r"""
    Loss of the agents'/clients' predictions.

    Table:
        Loss of the agents'/clients' final x.

    Plot:
        Loss (y-axis) per iteration (x-axis).

        Loss is calculated as the mean loss across agents/clients, where each agent's/client's loss is calculated using
        its recorded x at that iteration.

    """

    description: str = "loss"

    def get_data_from_trial(  # noqa: D102
        self,
        network: NetworkMetricsView,
        _: "BenchmarkProblem",
        iteration: int,
    ) -> list[float]:
        return utils._losses(network.agents(), iteration)  # noqa: SLF001


def _requires_fednetwork(problem: "BenchmarkProblem", metric_name: str) -> tuple[bool, str | None]:
    if not isinstance(problem.network, FedNetwork):
        return False, f"{metric_name} only applies to FedNetwork"
    return True, None


def _server_metric_cost(network: NetworkMetricsView, metric_name: str) -> Cost:
    agent_views = network.agents()
    if not agent_views:
        raise ValueError(f"{metric_name} requires at least one client metrics view")
    return agent_views[0].cost


class ClientDriftFromServer(Metric):
    r"""
    Distance between client local models and the server model.

    Table:
        Distance of the clients' final states from the final server state.

    Plot:
        Client drift from server (y-axis) per iteration (x-axis).

    The client drift per client is defined as:

    .. math::
        \{ \|\mathbf{x}_i - \mathbf{x}_s\|, \|\mathbf{x}_j - \mathbf{x}_s\|, ... \}

    where :math:`\mathbf{x}_s` is the current server state.

    Note:
        Available only for :class:`~decent_bench.networks.FedNetwork`.

    """

    description: str = "client drift from server"

    def is_available(  # noqa: D102
        self,
        problem: "BenchmarkProblem",
    ) -> tuple[bool, str | None]:
        return _requires_fednetwork(problem, self.description)

    def get_data_from_trial(  # noqa: D102
        self,
        network: NetworkMetricsView,
        problem: "BenchmarkProblem",  # noqa: ARG002
        iteration: int,
    ) -> list[float]:
        return utils._drifts(network.clients(), network.server(), iteration)  # noqa: SLF001

    def get_plot_data(
        self,
        network: NetworkMetricsView,
        _: "BenchmarkProblem",
    ) -> list[tuple[float, float]]:
        """Extract client drift trajectory."""
        server = network.server()
        agent_views = network.clients()
        return [
            (i, float(np.mean(utils._drifts(agent_views, server, i))))  # noqa: SLF001
            for i in utils.all_sorted_iterations([server])
        ]


class FractionSelectedClients(Metric):
    r"""
    Fraction of clients selected by the federated algorithm to perform local training.

    Table:
        Fraction of selected clients over the algorithm run.

    Note:
        Available only for :class:`~decent_bench.networks.FedNetwork`.

    """

    description: str = "fraction selected clients"

    def is_available(  # noqa: D102
        self,
        problem: "BenchmarkProblem",
    ) -> tuple[bool, str | None]:
        return _requires_fednetwork(problem, self.description)

    def get_data_from_trial(  # noqa: D102
        self,
        network: NetworkMetricsView,
        _: "BenchmarkProblem",
        __: int,
    ) -> tuple[float]:
        agent_views = network.clients()
        n_rounds = utils._observed_rounds(agent_views)  # noqa: SLF001
        if n_rounds == 0 or not agent_views:
            return (np.nan,)
        return (sum(agent.n_times_selected for agent in agent_views) / (n_rounds * len(agent_views)),)


class ServerMSE(Metric):
    r"""
    Mean squared error of the server model's predictions.

    Table:
        Mean squared error of the final server x.

    Plot:
        Server MSE (y-axis) per iteration (x-axis).

    Note:
        Available only for :class:`~decent_bench.networks.FedNetwork` with ``problem.test_data`` and empirical-risk
        client costs.

    """

    description: str = "server mse"

    def is_available(  # noqa: D102
        self,
        problem: "BenchmarkProblem",
    ) -> tuple[bool, str | None]:
        available, reason = _requires_fednetwork(problem, self.description)
        if not available:
            return False, reason
        if getattr(problem, "test_data", None) is None:
            return False, "requires problem.test_data"
        if not all(isinstance(a.cost, EmpiricalRiskCost) for a in problem.network.agents()):
            return False, "server MSE only applies if all clients have EmpiricalRiskCost"
        return True, None

    def get_data_from_trial(  # noqa: D102
        self,
        network: NetworkMetricsView,
        problem: "BenchmarkProblem",
        iteration: int,
    ) -> tuple[float]:
        cost = _server_metric_cost(network, self.description)
        return (utils._mse_at_x(cost, network.server().x_history[iteration], problem),)  # noqa: SLF001

    def get_plot_data(
        self,
        network: NetworkMetricsView,
        problem: "BenchmarkProblem",
    ) -> list[tuple[float, float]]:
        """Extract server MSE trajectory."""
        server = network.server()
        return [(i, self.get_data_from_trial(network, problem, i)[0]) for i in utils.all_sorted_iterations([server])]


class ServerAccuracy(Metric):
    r"""
    Accuracy of the server model's predictions.

    Table:
        Accuracy of the final server x.

    Plot:
        Server accuracy (y-axis) per iteration (x-axis).

    Note:
        Available only for :class:`~decent_bench.networks.FedNetwork` with ``problem.test_data``, empirical-risk
        client costs, and integer-valued targets.

    """

    description: str = "server accuracy"

    def is_available(  # noqa: D102
        self,
        problem: "BenchmarkProblem",
    ) -> tuple[bool, str | None]:
        available, reason = _requires_fednetwork(problem, self.description)
        if not available:
            return False, reason
        if getattr(problem, "test_data", None) is None:
            return False, "requires problem.test_data"
        if not all(isinstance(a.cost, EmpiricalRiskCost) for a in problem.network.agents()):
            return False, "server accuracy only applies if all clients have EmpiricalRiskCost"
        _, test_y = utils._split_dataset(problem.test_data)  # type: ignore[arg-type]  # noqa: SLF001
        if test_y.dtype.kind not in {"i", "u"}:
            return False, f"server accuracy only applies for integer targets, dtype {test_y.dtype} found"
        return True, None

    def get_data_from_trial(  # noqa: D102
        self,
        network: NetworkMetricsView,
        problem: "BenchmarkProblem",
        iteration: int,
    ) -> tuple[float]:
        cost = _server_metric_cost(network, self.description)
        return (utils._accuracy_at_x(cost, network.server().x_history[iteration], problem),)  # noqa: SLF001

    def get_plot_data(
        self,
        network: NetworkMetricsView,
        problem: "BenchmarkProblem",
    ) -> list[tuple[float, float]]:
        """Extract server accuracy trajectory."""
        server = network.server()
        return [(i, self.get_data_from_trial(network, problem, i)[0]) for i in utils.all_sorted_iterations([server])]


_BASE_TABLE_METRICS: list[Metric] = [
    Regret(),
    GradientNorm(),
    XError(),
    ConsensusError([min, np.average, max]),
    Loss([min, np.average, max]),
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
- :class:`Regret` (no statistics)
- :class:`GradientNorm` (no statistics)
- :class:`XError` (no statistics)
- :class:`ConsensusError` - :func:`min`, :func:`~numpy.average`, :func:`max`
- :class:`Loss` - :func:`min`, :func:`~numpy.average`, :func:`max`
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

_REGRESSION_TABLE_METRICS: list[Metric] = [
    MSE([min, np.average, max], x_log=False, y_log=True),
]
"""
- :class:`MSE` - :func:`min`, :func:`~numpy.average`, :func:`max`

:meta hide-value:
"""

_CLASSIFICATION_TABLE_METRICS: list[Metric] = [
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
_BASE_PLOT_METRICS: list[Metric] = [
    Regret([], x_log=False, y_log=True),
    GradientNorm([], x_log=False, y_log=True),
    ConsensusError([], x_log=False, y_log=True),
    Loss([], x_log=False, y_log=False),
]
"""
- :class:`Regret` (semi-log)
- :class:`GradientNorm` (semi-log)
- :class:`ConsensusError` (semi-log)
- :class:`Loss` (linear)

:meta hide-value:
"""


_REGRESSION_PLOT_METRICS: list[Metric] = _REGRESSION_TABLE_METRICS
"""
- :class:`MSE` (semi-log)

:meta hide-value:
"""

_CLASSIFICATION_PLOT_METRICS: list[Metric] = _CLASSIFICATION_TABLE_METRICS
"""
- :class:`Accuracy` (linear)
- :class:`Precision` (linear)
- :class:`Recall` (linear)

:meta hide-value:
"""

_FEDERATED_TABLE_METRICS: list[Metric] = [
    ClientDriftFromServer([min, np.average, max]),
    FractionSelectedClients(fmt=".2%", x_log=False, y_log=False),
]
"""
- :class:`ClientDriftFromServer` - min, average, max
- :class:`FractionSelectedClients` - single value with percentage format

:meta hide-value:
"""

_FEDERATED_PLOT_METRICS: list[Metric] = [
    ClientDriftFromServer([], x_log=False, y_log=True),
]
"""
- :class:`ClientDriftFromServer` (semi-log)

:meta hide-value:
"""

_FEDERATED_REGRESSION_TABLE_METRICS: list[Metric] = [
    ServerMSE(x_log=False, y_log=True),
]
"""
- :class:`ServerMSE` - single value

:meta hide-value:
"""

_FEDERATED_REGRESSION_PLOT_METRICS: list[Metric] = [
    ServerMSE([], x_log=False, y_log=True),
]
"""
- :class:`ServerMSE` (semi-log)

:meta hide-value:
"""

_FEDERATED_CLASSIFICATION_TABLE_METRICS: list[Metric] = [
    ServerAccuracy(fmt=".2%", x_log=False, y_log=False),
]
"""
- :class:`ServerAccuracy` - single value with percentage format

:meta hide-value:
"""

_FEDERATED_CLASSIFICATION_PLOT_METRICS: list[Metric] = [
    ServerAccuracy([], fmt=".2%", x_log=False, y_log=False),
]
"""
- :class:`ServerAccuracy` (linear)

:meta hide-value:
"""

_DEFAULT_TABLE_METRICS: list[Metric] = [
    *_BASE_TABLE_METRICS,
    *_REGRESSION_TABLE_METRICS,
    *_CLASSIFICATION_TABLE_METRICS,
    *_FEDERATED_TABLE_METRICS,
    *_FEDERATED_REGRESSION_TABLE_METRICS,
    *_FEDERATED_CLASSIFICATION_TABLE_METRICS,
]

_DEFAULT_PLOT_METRICS: list[Metric] = [
    *_BASE_PLOT_METRICS,
    *_REGRESSION_PLOT_METRICS,
    *_CLASSIFICATION_PLOT_METRICS,
    *_FEDERATED_PLOT_METRICS,
    *_FEDERATED_REGRESSION_PLOT_METRICS,
    *_FEDERATED_CLASSIFICATION_PLOT_METRICS,
]
