"""Collection of pre-defined runtime metrics."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import decent_bench.metrics.metric_utils as utils
import decent_bench.utils.interoperability as iop
from decent_bench.metrics._runtime_metric import RuntimeMetric

if TYPE_CHECKING:
    from decent_bench.agents import Agent
    from decent_bench.benchmark_problem import BenchmarkProblem


class RuntimeLoss(RuntimeMetric):
    r"""
    Runtime loss metric.

    Computes the average loss across all agents at each iteration.
    This is useful for monitoring convergence and detecting issues early.

    The loss is computed as:

    .. math::
        \text{loss} = \frac{1}{N} \sum_{i=1}^{N} f_i(\mathbf{x}_i)

    where :math:`N` is the number of agents, :math:`f_i` is agent i's cost function,
    and :math:`\mathbf{x}_i` is agent i's current optimization variable.

    """

    description = "Loss"

    def compute(self, _: "BenchmarkProblem", agents: Sequence["Agent"], __: int) -> float:  # noqa: D102
        total_loss = sum(agent.cost.function(agent.x) for agent in agents)
        return total_loss / len(agents)


class RuntimeRegret(RuntimeMetric):
    r"""
    Runtime regret metric.

    Requires that the benchmark problem :attr:`~decent_bench.benchmark_problem.BenchmarkProblem.x_optimal` is defined.

    Regret is computed as:

    .. math::
        \text{regret} = \frac{1}{N} \sum_{i=1}^{N} f_i(\mathbf{x}_i) - \frac{1}{N} \sum_{i=1}^{N} f_i(\mathbf{x}^*)

    where :math:`N` is the number of agents, :math:`f_i` is agent i's cost function, :math:`\mathbf{x}_i` is agent i's
    current optimization variable, and :math:`\mathbf{x}^*` is the optimal solution.

    """

    description = "Regret"

    def compute(self, problem: "BenchmarkProblem", agents: Sequence["Agent"], _: int) -> float:  # noqa: D102
        if problem.x_optimal is None:
            return float("nan")

        agent_cost = sum(agent.cost.function(agent.x) for agent in agents) / len(agents)
        # If all agents have the same cost function, we can compute regret using the optimal value
        if utils.check_same_cost_functions(tuple(agent.cost for agent in agents)):
            optimal_cost = agents[0].cost.function(problem.x_optimal)
        else:
            optimal_cost = sum(agent.cost.function(problem.x_optimal) for agent in agents) / len(agents)
        return agent_cost - optimal_cost


class RuntimeGradientNorm(RuntimeMetric):
    r"""
    Runtime gradient norm metric.

    Computes the average gradient norm across all agents at each iteration.
    This is useful for monitoring if the algorithm is making progress towards
    a stationary point.

    The gradient norm is computed as:

    .. math::
        \text{grad_norm} = \frac{1}{N} \sum_{i=1}^{N} \|\nabla f_i(\mathbf{x}_i)\|

    where :math:`N` is the number of agents, :math:`f_i` is agent i's cost function,
    and :math:`\mathbf{x}_i` is agent i's current optimization variable.

    """

    description = "Gradient Norm"

    def compute(self, _: "BenchmarkProblem", agents: Sequence["Agent"], __: int) -> float:  # noqa: D102
        grad_norms = sum(float(iop.norm(agent.cost.gradient(agent.x))) for agent in agents)
        return grad_norms / len(agents)


class RuntimeConsensusError(RuntimeMetric):
    r"""
    Monitors how well agents agree on their decision variables.

    This is useful for diagnosing issues in decentralized algorithms where agents are supposed to reach consensus.

    The consensus error is computed as:

    .. math::
        \text{consensus_error} = \frac{1}{N} \sum_{i=1}^{N} \|\mathbf{x}_i - \bar{\mathbf{x}}\|

    where :math:`N` is the number of agents, :math:`\mathbf{x}_i` is agent i's current optimization variable,
    and :math:`\bar{\mathbf{x}}` is the average of all agents' optimization variables.

    """

    description = "Consensus Error"

    def compute(self, _: "BenchmarkProblem", agents: Sequence["Agent"], __: int) -> float:  # noqa: D102
        # Compute average x across all agents
        x_avg = iop.mean(iop.stack([agent.x for agent in agents]), dim=0)

        # Compute average distance from the mean
        errors = [float(iop.norm(agent.x - x_avg)) for agent in agents]
        return sum(errors) / len(agents)
