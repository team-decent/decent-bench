from collections.abc import Sequence
from functools import cache

import numpy as np
from numpy import float64
from numpy import linalg as la
from numpy.linalg import LinAlgError
from numpy.typing import NDArray
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
from sklearn import metrics as sk_metrics

import decent_bench.utils.interoperability as iop
from decent_bench import costs
from decent_bench.agents import AgentMetricsView
from decent_bench.benchmark_problem import BenchmarkProblem
from decent_bench.utils.array import Array
from decent_bench.utils.logger import LOGGER
from decent_bench.utils.types import Dataset


class MetricProgressBar(Progress):
    """
    Progress bar for metric calculations.

    Make sure to set the field *status* in the task to show custom status messages.

    """

    def __init__(self) -> None:
        super().__init__(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(elapsed_when_finished=True),
            TextColumn("{task.fields[status]}"),
        )


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
def x_mean(agents: tuple[AgentMetricsView, ...], iteration: int = -1) -> Array:
    """
    Calculate the mean x at *iteration* (or using the agents' final x if *iteration* is -1).

    Agents that did not reach *iteration* are disregarded.

    Raises:
        ValueError: if no agent reached *iteration*

    """
    if iteration == -1:
        all_x_at_iter = [a.x_history[max(a.x_history)] for a in agents if len(a.x_history) > 0]
    else:
        all_x_at_iter = [a.x_history[find_closest_iteration(a, iteration)] for a in agents]

    if len(all_x_at_iter) == 0:
        raise ValueError(f"No agent reached iteration {iteration}")

    return iop.mean(iop.stack(all_x_at_iter), dim=0)


def regret(agents: Sequence[AgentMetricsView], problem: BenchmarkProblem, iteration: int = -1) -> float:
    r"""
    Calculate the global regret at *iteration* (or using the agents' final x if *iteration* is -1).

    Global regret is defined as:

    .. include:: snippets/global_cost_error.rst
    """
    if problem.x_optimal is None:
        return float("nan")

    x_opt = problem.x_optimal
    mean_x = x_mean(tuple(agents), iteration)
    optimal_cost = sum(a.cost.function(x_opt) for a in agents)
    actual_cost = sum(a.cost.function(mean_x) for a in agents)
    return actual_cost - optimal_cost


def gradient_norm(agents: Sequence[AgentMetricsView], iteration: int = -1) -> float:
    r"""
    Calculate the global gradient norm at *iteration* (or using the agents' final x if *iteration* is -1).

    Global gradient norm is defined as:

    .. include:: snippets/global_gradient_optimality.rst
    """
    mean_x = x_mean(tuple(agents), iteration)
    grad_avg = sum(iop.to_numpy(a.cost.gradient(mean_x)) for a in agents) / len(agents)
    return float(la.norm(grad_avg)) ** 2


@cache
def x_error(agent: AgentMetricsView, problem: BenchmarkProblem, up_to_iteration: int) -> NDArray[float64]:
    r"""
    Calculate the x error per iteration as defined below (until up_to_iteration iteration).

    If *up_to_iteration* is -1, all iterations are taken into account. Otherwise,
    only iterations up to and including *up_to_iteration* are taken into account, subsequent iterations are disregarded.

    .. math::
        \{ \|\mathbf{x}_0 - \mathbf{x}^\star\|, \|\mathbf{x}_1 - \mathbf{x}^\star\|, ... \}

    where :math:`\mathbf{x}_k` is the agent's local x at iteration k,
    and :math:`\mathbf{x}^\star` is the optimal x defined in the *problem*.
    """
    if up_to_iteration == -1:
        up_to_iteration = int(1e100)

    if problem.x_optimal is None:
        return np.array([np.nan for iteration, _ in sorted(agent.x_history.items()) if iteration <= up_to_iteration])

    x_per_iteration = np.asarray([
        iop.to_numpy(x) for iteration, x in sorted(agent.x_history.items()) if iteration <= up_to_iteration
    ])
    opt_x = iop.to_numpy(problem.x_optimal)
    errors: NDArray[float64] = la.norm(x_per_iteration - opt_x, axis=tuple(range(1, x_per_iteration.ndim)))
    return errors


@cache
def asymptotic_convergence_rate_and_order(
    agent: AgentMetricsView,
    problem: BenchmarkProblem,
    up_to_iteration: int,
) -> tuple[float, float]:
    r"""
    Estimate the asymptotic convergence rate and order as defined below (until up_to_iteration iteration).

    If *up_to_iteration* is -1, all iterations are taken into account. Otherwise,
    only iterations up to and including *up_to_iteration* are taken into account, subsequent iterations are disregarded.

    .. include:: snippets/asymptotic_convergence_rate_and_order.rst
    """
    errors = x_error(agent, problem, up_to_iteration)
    if not np.isfinite(errors).all():
        return np.nan, np.nan
    errors = errors[errors > 0]
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
def iterative_convergence_rate_and_order(
    agent: AgentMetricsView,
    problem: BenchmarkProblem,
    up_to_iteration: int,
) -> tuple[float, float]:
    r"""
    Estimate the iterative convergence rate and order as defined below (until up_to_iteration iteration).

    If *up_to_iteration* is -1, all iterations are taken into account. Otherwise,
    only iterations up to and including *up_to_iteration* are taken into account, subsequent iterations are disregarded.

    .. include:: snippets/iterative_convergence_rate_and_order.rst
    """
    errors = x_error(agent, problem, up_to_iteration)
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


def accuracy(agents: Sequence[AgentMetricsView], problem: BenchmarkProblem, iteration: int) -> list[float]:
    """
    Calculate the accuracy per agent.

    Accuracy is only applicable for problems using :class:`~decent_bench.costs.EmpiricalRiskCost`.

    Args:
        agents: sequence of agents to calculate accuracy for
        problem: benchmark problem containing test data
        iteration: iteration to calculate accuracy at, or -1 to use the agents' final x

    Returns:
        list of accuracies per agent at *iteration*

    """
    if problem.test_data is None:
        LOGGER.warning(
            "Test data is required to calculate accuracy but is not provided in the problem, returning NaN for accuracy"
        )
        return [np.nan for _ in agents]

    if not all(isinstance(a.cost, costs.EmpiricalRiskCost) for a in agents):
        LOGGER.warning(
            "Accuracy metric is only applicable for EmpiricalRiskCost, but at least one agent has a different cost, "
            "returning NaN for accuracy"
        )
        return [np.nan for _ in agents]

    _, test_y = split_dataset(problem.test_data)

    if test_y.dtype.kind not in {"i", "u"}:
        LOGGER.warning(
            "Accuracy calculation is only applicable for integer targets, but "
            f"targets have values of dtype {test_y.dtype}, returning NaN for accuracy"
        )
        return [np.nan for _ in agents]

    ret: list[float] = []
    for agent in agents:
        if isinstance(agent.cost, costs.EmpiricalRiskCost):
            agent_iteration = max(agent.x_history) if iteration == -1 else iteration
            agent_iteration = find_closest_iteration(agent, agent_iteration)
            preds = predict_agent(agent, agent_iteration, problem)
            ret.append(float(sk_metrics.accuracy_score(test_y, preds)))
        else:
            ret.append(np.nan)
    return ret


def mse(agents: Sequence[AgentMetricsView], problem: BenchmarkProblem, iteration: int) -> list[float]:
    """
    Calculate the mean squared error (MSE) per agent.

    MSE is only applicable for problems using :class:`~decent_bench.costs.EmpiricalRiskCost`.

    Args:
        agents: sequence of agents to calculate MSE for
        problem: benchmark problem containing test data
        iteration: iteration to calculate MSE at, or -1 to use the agents' final x

    Returns:
        list of MSE per agent

    """
    if problem.test_data is None:
        LOGGER.warning(
            "Test data is required to calculate MSE but is not provided in the problem, returning NaN for MSE"
        )
        return [np.nan for _ in agents]

    if not all(isinstance(a.cost, costs.EmpiricalRiskCost) for a in agents):
        LOGGER.warning(
            "MSE metric is only applicable for EmpiricalRiskCost, but at least one agent has a different cost, "
            "returning NaN for MSE"
        )
        return [np.nan for _ in agents]

    ret: list[float] = []
    _, test_y = split_dataset(problem.test_data)
    for agent in agents:
        if isinstance(agent.cost, costs.EmpiricalRiskCost):
            agent_iteration = max(agent.x_history) if iteration == -1 else iteration
            agent_iteration = find_closest_iteration(agent, agent_iteration)
            preds = predict_agent(agent, agent_iteration, problem)
            ret.append(sk_metrics.mean_squared_error(test_y, preds))
        else:
            ret.append(np.nan)
    return ret


def precision(agents: Sequence[AgentMetricsView], problem: BenchmarkProblem, iteration: int) -> list[float]:
    """
    Calculate the precision per agent.

    Precision is only applicable for problems using :class:`~decent_bench.costs.EmpiricalRiskCost`.
    Calculated using :func:`sklearn.metrics.precision_score` with micro averaging.

    Args:
        agents: sequence of agents to calculate precision for
        problem: benchmark problem containing test data
        iteration: iteration to calculate precision at, or -1 to use the agents' final x

    Returns:
        list of precision per agent at *iteration*

    """
    if problem.test_data is None:
        LOGGER.warning(
            "Test data is required to calculate precision but is not provided "
            "in the problem, returning NaN for precision"
        )
        return [np.nan for _ in agents]

    if not all(isinstance(a.cost, costs.EmpiricalRiskCost) for a in agents):
        LOGGER.warning(
            "Precision metric is only applicable for EmpiricalRiskCost, but at least one agent has a different cost, "
            "returning NaN for precision"
        )
        return [np.nan for _ in agents]

    _, test_y = split_dataset(problem.test_data)

    if test_y.dtype.kind not in {"i", "u"}:
        LOGGER.warning(
            "Precision calculation is only applicable for integer targets, but "
            f"targets have values of dtype {test_y.dtype}, returning NaN for precision"
        )
        return [np.nan for _ in agents]

    ret: list[float] = []
    for agent in agents:
        if isinstance(agent.cost, costs.EmpiricalRiskCost):
            agent_iteration = max(agent.x_history) if iteration == -1 else iteration
            agent_iteration = find_closest_iteration(agent, agent_iteration)
            preds = predict_agent(agent, agent_iteration, problem)
            ret.append(float(sk_metrics.precision_score(test_y, preds, average="micro")))
        else:
            ret.append(np.nan)
    return ret


def recall(agents: Sequence[AgentMetricsView], problem: BenchmarkProblem, iteration: int) -> list[float]:
    """
    Calculate the recall per agent.

    Recall is only applicable for problems using :class:`~decent_bench.costs.EmpiricalRiskCost`.
    Calculated using :func:`sklearn.metrics.recall_score` with micro averaging.

    Args:
        agents: sequence of agents to calculate recall for
        problem: benchmark problem containing test data
        iteration: iteration to calculate recall at, or -1 to use the agents' final x

    Returns:
        list of recall per agent at *iteration*

    """
    if problem.test_data is None:
        LOGGER.warning(
            "Test data is required to calculate recall but is not provided in the problem, returning NaN for recall"
        )
        return [np.nan for _ in agents]

    if not all(isinstance(a.cost, costs.EmpiricalRiskCost) for a in agents):
        LOGGER.warning(
            "Recall metric is only applicable for EmpiricalRiskCost, but at least one agent has a different cost, "
            "returning NaN for recall"
        )
        return [np.nan for _ in agents]

    _, test_y = split_dataset(problem.test_data)

    if test_y.dtype.kind not in {"i", "u"}:
        LOGGER.warning(
            "Recall calculation is only applicable for integer targets, but "
            f"targets have values of dtype {test_y.dtype}, returning NaN for recall"
        )
        return [np.nan for _ in agents]

    ret: list[float] = []
    for agent in agents:
        if isinstance(agent.cost, costs.EmpiricalRiskCost):
            agent_iteration = max(agent.x_history) if iteration == -1 else iteration
            agent_iteration = find_closest_iteration(agent, agent_iteration)
            preds = predict_agent(agent, agent_iteration, problem)
            ret.append(float(sk_metrics.recall_score(test_y, preds, average="micro")))
        else:
            ret.append(np.nan)
    return ret


def split_dataset(data: Dataset) -> tuple[tuple[Array, ...], NDArray[float64]]:
    """
    Split dataset into features and labels.

    Args:
        data: dataset to split, as a tuple of (features, labels)

    Returns:
        tuple of (features, labels)

    """
    x, y = zip(*data, strict=True)
    test_x = tuple(x)
    test_y = np.array(y)
    return test_x, test_y


@cache
def predict_agent(agent: AgentMetricsView, iteration: int, problem: BenchmarkProblem) -> NDArray[float64]:
    """
    Get the predictions of *agent* at *iteration* on *test_x*.

    This function is cached since predictions can be expensive to compute and are used in multiple metrics.

    Args:
        agent: agent to get predictions from
        iteration: iteration to get predictions at
        problem: benchmark problem containing test data

    Returns:
        predictions of *agent* at *iteration* on *test_x*

    Raises:
        TypeError: if *agent* does not have an :class:`~decent_bench.costs.EmpiricalRiskCost` cost
        ValueError: if *problem* does not have test data

    Note:
        Make sure the *iteration* is present in the agent's x_history.

    """
    if not isinstance(agent.cost, costs.EmpiricalRiskCost):
        raise TypeError("Predictions can only be obtained for agents with EmpiricalRiskCost")

    if problem.test_data is None:
        raise ValueError("Test data is required to get predictions but is not provided in the problem")

    test_x, _ = split_dataset(problem.test_data)

    return iop.to_numpy(agent.cost.predict(agent.x_history[iteration], list(test_x)))


@cache
def find_closest_iteration(agent: AgentMetricsView, target_iteration: int) -> int:
    """
    Find the most recent iteration in *agent* that is <= *target_iteration*.

    If *target_iteration* is in *agent*'s x_history then it is returned.
    Otherwise, the most recent iteration in *agent*'s x_history that is <= *target_iteration* is returned.
    If no iteration is <= *target_iteration*, the earliest iteration is returned.

    Args:
        agent: agent to find the iteration in
        target_iteration: iteration to find the most recent iteration <= to

    Returns:
        most recent iteration in *agent* that is <= *target_iteration*

    """
    if target_iteration in agent.x_history:
        return target_iteration
    iterations = np.array(sorted(agent.x_history.keys()))
    valid_iterations = iterations[iterations <= target_iteration]
    if len(valid_iterations) == 0:
        # No iteration <= target_iteration, return the first iteration
        # This should not occur as we always include iteration 0 in the x_history, but we include this for safety
        return int(iterations[0])
    return int(valid_iterations[-1])


def common_sorted_iterations(agents: Sequence[AgentMetricsView]) -> list[int]:
    """
    Get a sorted list of all common iterations reached by agents in *agents*.

    Since the agents can sample their states periodically, and may sample at different iterations,
    this function returns only the iterations that are common to all agents. These iterations can then be used
    to compute metrics that require synchronized iterations.

    Args:
        agents: sequence of agents to get the common iterations from

    Returns:
        sorted list of iterations reached by all agents

    """
    common_iters = set.intersection(*(set(a.x_history.keys()) for a in agents)) if agents else set()
    return sorted(common_iters)


def all_sorted_iterations(agents: Sequence[AgentMetricsView]) -> list[int]:
    """
    Get a sorted list of all iterations reached by any agent in *agents*.

    Since the agents can sample their states periodically, and may sample at different iterations,
    this function returns the union of all iterations reached by any agent. These iterations can then be used
    to compute metrics that do not require synchronized iterations, with missing samples for agents that did not
    sample at those iterations replaced by the most recent previous sample for that agent.

    Args:
        agents: sequence of agents to get the iterations from

    Returns:
        sorted list of iterations reached by any agent

    """
    all_iters = set.union(*(set(a.x_history.keys()) for a in agents)) if agents else set()
    return sorted(all_iters)


@cache
def check_same_cost_functions(costs: tuple[costs.Cost]) -> bool:
    """
    Check if all costs in *costs* have the same function.

    This is useful for metrics that can be computed more efficiently if all agents have the same cost function.

    Args:
        costs: sequence of costs to check

    Returns:
        True if all costs have the same function, False otherwise

    """
    if len(costs) == 0:
        return True
    first_function = costs[0]
    return all(cost == first_function for cost in costs)
