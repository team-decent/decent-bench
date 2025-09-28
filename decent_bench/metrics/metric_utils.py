from collections.abc import Sequence
from functools import cache

import numpy as np
from numpy import float64
from numpy import linalg as la
from numpy.linalg import LinAlgError
from numpy.typing import NDArray

from decent_bench.agent import AgentMetricsView
from decent_bench.benchmark_problem import BenchmarkProblem


def single(values: Sequence[float]) -> float:
    """
    Assert that *values* contain exactly one element and return it.

    Raises:
        ValueError: if there isn't exactly one element in *values*

    """
    if len(values) != 1:
        raise ValueError("Argument `values` must have exactly 1 element")
    return values[0]


@cache
def mean_x(agents: tuple[AgentMetricsView, ...], iteration: int = -1) -> NDArray[float64]:
    """
    Calculate the mean x at *iteration* (or using the agents' final x if *iteration* is -1).

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
    Calculate the global cost error at *iteration* (or using the agents' final x if *iteration* is -1).

    Global cost error is defined as:

    .. include:: snippets/global_cost_error.rst
    """
    x_opt = problem.optimal_x
    x_mean = mean_x(tuple(agents), iteration)
    optimal_cost = sum(a.cost_function.evaluate(x_opt) for a in agents)
    actual_cost = sum(a.cost_function.evaluate(x_mean) for a in agents)
    return abs(optimal_cost - actual_cost)


def global_gradient_optimality_at_iter(agents: list[AgentMetricsView], iteration: int = -1) -> float:
    r"""
    Calculate the global gradient optimality at *iteration* (or using the agents' final x if *iteration* is -1).

    Global gradient optimality is defined as:

    .. include:: snippets/global_gradient_optimality.rst
    """
    x_mean = mean_x(tuple(agents), iteration)
    grad_avg = sum(a.cost_function.gradient(x_mean) for a in agents) / len(agents)
    return float(la.norm(grad_avg)) ** 2


@cache
def x_error_per_iteration(agent: AgentMetricsView, problem: BenchmarkProblem) -> NDArray[float64]:
    r"""
    Calculate the x error per iteration as defined below.

    .. math::
        \{ \|\mathbf{x}_0 - \mathbf{x}^\star\|, \|\mathbf{x}_1 - \mathbf{x}^\star\|, ... \}

    where :math:`\mathbf{x}_k` is the agent's local x at iteration k,
    and :math:`\mathbf{x}^\star` is the optimal x defined in the *problem*.
    """
    x_per_iteration = np.asarray(agent.x_per_iteration)
    opt_x = problem.optimal_x
    errors: NDArray[float64] = la.norm(x_per_iteration - opt_x, axis=tuple(range(1, x_per_iteration.ndim)))
    return errors


@cache
def asymptotic_convergence_rate_and_order(agent: AgentMetricsView, problem: BenchmarkProblem) -> tuple[float, float]:
    r"""
    Estimate the asymptotic convergence rate and order as defined below.

    .. include:: snippets/asymptotic_convergence_rate_and_order.rst
    """
    errors = x_error_per_iteration(agent, problem)
    errors = errors[errors > 0]
    if not np.isfinite(errors).all():
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


@cache
def iterative_convergence_rate_and_order(agent: AgentMetricsView, problem: BenchmarkProblem) -> tuple[float, float]:
    r"""
    Estimate the iterative convergence rate and order as defined below.

    .. include:: snippets/iterative_convergence_rate_and_order.rst
    """
    errors = x_error_per_iteration(agent, problem)
    if not np.isfinite(errors).all():
        return np.nan, np.nan
    iterations_and_errors = [(i + 1, e) for i, e in enumerate(errors)]
    iterations_and_errors = [ie for ie in iterations_and_errors if ie[1] > 0]
    log_errors = np.log([ie[1] for ie in iterations_and_errors])
    log_iterations = np.log([ie[0] for ie in iterations_and_errors])
    try:
        slope, intercept = np.polyfit(log_errors, log_iterations, 1)
        rate, order = np.exp(intercept), -slope
    except LinAlgError:
        rate, order = np.nan, np.nan
    return rate, order
