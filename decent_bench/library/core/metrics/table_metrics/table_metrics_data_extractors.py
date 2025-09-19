from numpy import linalg as la

import decent_bench.library.core.metrics.metric_utils as utils
from decent_bench.library.core.agent import AgentMetricsView
from decent_bench.library.core.benchmark_problem.benchmark_problems import BenchmarkProblem


def global_cost_error(agents: list[AgentMetricsView], problem: BenchmarkProblem) -> tuple[float]:
    """
    Calculate the global cost error using the agents' final x.

    Global cost error is defined as:

    .. include:: snippets/global_cost_error.rst
    """
    return (utils.global_cost_error_at_iter(agents, problem, iteration=-1),)


def global_gradient_optimality(agents: list[AgentMetricsView], _: BenchmarkProblem) -> tuple[float]:
    """
    Calculate the global gradient optimality using the agents' final x.

    Global gradient optimality is defined as:

    .. include:: snippets/global_gradient_optimality.rst
    """
    return (utils.global_gradient_optimality_at_iter(agents, iteration=-1),)


def x_error(agents: list[AgentMetricsView], problem: BenchmarkProblem) -> list[float]:
    r"""
    Calculate the x error per agent as defined below.

    .. math::
        \{ \|\mathbf{x}_i - \mathbf{x}^\star\|, \|\mathbf{x}_j - \mathbf{x}^\star\|, ... \}

    where :math:`\mathbf{x}_i` is agent i's final x,
    :math:`\mathbf{x}_j` is agent j's final x,
    and :math:`\mathbf{x}^\star` is the optimal x defined in the *problem*.

    """
    return [float(la.norm(problem.optimal_x - a.x_per_iteration[-1])) for a in agents]


def asymptotic_convergence_rate(agents: list[AgentMetricsView], problem: BenchmarkProblem) -> list[float]:
    """
    Estimate the asymptotic convergence rate per agent as defined below.

    .. include:: snippets/asymptotic_convergence_rate_and_order.rst
    """
    return [utils.asymptotic_convergence_rate_and_order(a, problem)[0] for a in agents]


def asymptotic_convergence_order(agents: list[AgentMetricsView], problem: BenchmarkProblem) -> list[float]:
    """
    Estimate the asymptotic convergence order per agent as defined below.

    .. include:: snippets/asymptotic_convergence_rate_and_order.rst
    """
    return [utils.asymptotic_convergence_rate_and_order(a, problem)[1] for a in agents]


def iterative_convergence_rate(agents: list[AgentMetricsView], problem: BenchmarkProblem) -> list[float]:
    """
    Estimate the iterative convergence rate per agent as defined below.

    .. include:: snippets/iterative_convergence_rate_and_order.rst
    """
    return [utils.iterative_convergence_rate_and_order(a, problem)[0] for a in agents]


def iterative_convergence_order(agents: list[AgentMetricsView], problem: BenchmarkProblem) -> list[float]:
    """
    Estimate the iterative convergence order per agent as defined below.

    .. include:: snippets/iterative_convergence_rate_and_order.rst
    """
    return [utils.iterative_convergence_rate_and_order(a, problem)[1] for a in agents]


def n_x_updates(agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[float]:
    """Get the number of iterations/updates of x per agent."""
    return [len(a.x_per_iteration) - 1 for a in agents]


def n_evaluate_calls(agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[float]:
    """Get the number of cost function evaluate calls per agent."""
    return [a.n_evaluate_calls for a in agents]


def n_gradient_calls(agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[float]:
    """Get the number of cost function gradient calls per agent."""
    return [a.n_gradient_calls for a in agents]


def n_hessian_calls(agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[float]:
    """Get the number of cost function hessian calls per agent."""
    return [a.n_hessian_calls for a in agents]


def n_proximal_calls(agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[float]:
    """Get the number of cost function proximal calls per agent."""
    return [a.n_proximal_calls for a in agents]


def n_sent_messages(agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[float]:
    """Get the number of sent messages per agent."""
    return [a.n_sent_messages for a in agents]


def n_received_messages(agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[float]:
    """Get the number of received messages per agent."""
    return [a.n_received_messages for a in agents]


def n_sent_messages_dropped(agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[float]:
    """Get the number of sent messages that were dropped per agent."""
    return [a.n_sent_messages_dropped for a in agents]
