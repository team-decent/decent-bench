from collections.abc import Sequence
from functools import lru_cache

import numpy as np
from numpy import float64
from numpy import linalg as la
from numpy.linalg import LinAlgError
from numpy.typing import NDArray

from decent_bench.library.core.agent import AgentMetricsView
from decent_bench.library.core.benchmark_problem.benchmark_problems import BenchmarkProblem


def single(values: Sequence[float]) -> float:
    """
    Assert that *values* contain exactly one element and return this element.

    Raises:
        ValueError: if there isn't exactly one element in *values*

    """
    if len(values) != 1:
        raise ValueError("Argument `values` must have exactly 1 element")
    return values[0]


@lru_cache
def mean_x(agents: tuple[AgentMetricsView, ...], iteration: int = -1) -> NDArray[float64]:
    """
    Calculate the mean x across all *agents* at a specific iteration (or -1 for the last iteration).

    Agents that did not reach *iteration* are disregarded.

    Raises:
        ValueError: if no agent reached *iteration*

    """
    all_x_at_iter = [a.x_per_iteration[iteration] for a in agents if len(a.x_per_iteration) > iteration]
    if len(all_x_at_iter) == 0:
        raise ValueError(f"No agent reached iteration {iteration}")
    res: NDArray[float64] = np.mean(all_x_at_iter, axis=0)
    return res


def global_cost_error_at_iter(agents: list[AgentMetricsView], problem: BenchmarkProblem, iteration: int = -1) -> float:
    r"""
    Calculate the global cost error as defined below.

    .. math::
        | \sum_i (f_i(\mathbf{\bar{x}}_k) - f_i(\mathbf{x}^\star)) |

    where i is an agent,
    :math:`f_i` is agent i's local cost function,
    :math:`\mathbf{\bar{x}}_k` is the mean x across all *agents* at iteration k,
    and :math:`\mathbf{x}^\star` is the optimal x defined in the *problem*.
    """
    x_opt = problem.optimal_x
    x_mean = mean_x(tuple(agents), iteration)
    optimal_cost = sum(a.cost_function.evaluate(x_opt) for a in agents)
    actual_cost = sum(a.cost_function.evaluate(x_mean) for a in agents)
    return abs(optimal_cost - actual_cost)


def global_gradient_optimality_at_iter(agents: list[AgentMetricsView], iteration: int = -1) -> float:
    r"""
    Calculate the global gradient optimality as defined below.

    .. math::
        \| \frac{1}{N} \sum_i \nabla f_i(\mathbf{\bar{x}}_k) \|^2

    where N is the number of *agents*,
    i is an agent,
    :math:`f_i` is agent i's local cost function,
    and :math:`\mathbf{\bar{x}}_k` is the mean x across all *agents* at iteration k.
    """
    x_mean = mean_x(tuple(agents), iteration=iteration)
    grad_avg = sum(a.cost_function.gradient(x_mean) for a in agents) / len(agents)
    return float(la.norm(grad_avg)) ** 2


@lru_cache
def x_error_per_iteration(agent: AgentMetricsView, problem: BenchmarkProblem) -> NDArray[float64]:
    r"""
    Calculate the agent's x error per iteration as defined below.

    .. math::
        \{ \|\mathbf{x}_0 - \mathbf{x}^\star\|, \|\mathbf{x}_1 - \mathbf{x}^\star\|, ... \}

    where :math:`\mathbf{x}_k` is the agent's local x at iteration k,
    and :math:`\mathbf{x}^\star` is the optimal x defined in the *problem*.
    """
    x_per_iteration = np.asarray(agent.x_per_iteration)
    opt_x = problem.optimal_x
    errors: NDArray[float64] = la.norm(x_per_iteration - opt_x, axis=tuple(range(1, x_per_iteration.ndim)))
    return errors


@lru_cache
def asymptotic_convergence_rate_and_order(agent: AgentMetricsView, problem: BenchmarkProblem) -> tuple[float, float]:
    r"""
    Estimate the asymptotic convergence rate :math:`\mu` and order :math:`q` as defined below.

    .. math::
        \lim_{k \to \infty}
        \frac{\| \mathbf{x}_{k+1} - \mathbf{x}^\star \|}{\| \mathbf{x}_{k} - \mathbf{x}^\star\|^q} = \mu

    where :math:`\mathbf{x}_k` is the agent's local x at iteration k,
    :math:`\mathbf{x}^\star` is the optimal x defined in the *problem*,
    :math:`q` is the asymptotic convergence order,
    and :math:`\mu` is the asymptotic convergence rate.
    """
    errors = x_error_per_iteration(agent, problem)
    errors = errors[errors > 0]
    if any(np.isinf(e) or np.isnan(e) for e in errors):
        return np.nan, np.nan
    log_errors = np.log(errors)
    x = log_errors[:-1]
    y = log_errors[1:]
    try:
        slope, intercept = np.polyfit(x, y, 1)
        rate, order = np.exp(intercept), slope
    except LinAlgError:
        rate, order = np.nan, np.nan
    return rate, order


@lru_cache
def iterative_convergence_rate_and_order(agent: AgentMetricsView, problem: BenchmarkProblem) -> tuple[float, float]:
    r"""
    Estimate the iterative convergence rate :math:`\mu` and order :math:`q` as defined below.

    .. math::
        k = \frac{\mu}{\|\mathbf{x}_k - \mathbf{x}^\star\|^q}

    where k is the iteration,
    :math:`\mu` is the iterative convergence rate,
    :math:`\mathbf{x}_k` is the agent's local x at iteration k,
    :math:`\mathbf{x}^\star` is the optimal x defined in the *problem*,
    and :math:`q` is the iterative convergence order.
    """
    errors = x_error_per_iteration(agent, problem)
    if any(np.isinf(e) or np.isnan(e) for e in errors):
        return np.nan, np.nan
    iterations_and_errors = [(i + 1, e) for i, e in enumerate(errors)]
    iterations_and_errors = [ie for ie in iterations_and_errors if ie[1] > 0]
    log_errors = np.log([e[1] for e in iterations_and_errors])
    log_iterations = np.log([e[0] for e in iterations_and_errors])
    try:
        slope, intercept = np.polyfit(log_errors, log_iterations, 1)
        rate, order = np.exp(intercept), -slope
    except LinAlgError:
        rate, order = np.nan, np.nan
    return rate, order
