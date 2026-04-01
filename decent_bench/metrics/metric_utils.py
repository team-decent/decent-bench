from collections.abc import Sequence
from functools import cache
from typing import TYPE_CHECKING

import numpy as np
from numpy import float64
from numpy import linalg as la
from numpy.typing import NDArray
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
from sklearn import metrics as sk_metrics

import decent_bench.utils.interoperability as iop
from decent_bench import costs
from decent_bench.agents import AgentMetricsView
from decent_bench.utils.array import Array
from decent_bench.utils.types import Dataset

if TYPE_CHECKING:
    from decent_bench.benchmark import BenchmarkProblem


class MetricUnavailableError(Exception):
    """
    Exception raised when a requested metric cannot be computed.

    For example when ``x_optimal`` is not available and :func:`regret` cannot be computed.

    Args:
        message: motivation for unavailability

    """

    def __init__(self, message: str):
        super().__init__(message)


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
        all_x_at_iter = [a.x_history[a.x_history.max()] for a in agents]
    else:
        all_x_at_iter = [a.x_history[iteration] for a in agents]

    if len(all_x_at_iter) == 0:
        raise ValueError(f"No agent reached iteration {iteration}")

    return iop.mean(iop.stack(all_x_at_iter), dim=0)


def regret(agents: Sequence[AgentMetricsView], problem: "BenchmarkProblem", iteration: int = -1) -> float:
    r"""
    Calculate the global regret at *iteration* (or using the agents' final x if *iteration* is -1).

    Global regret is defined as:

    .. include:: snippets/global_cost_error.rst

    Raises:
        MetricUnavailableError: if ``problem.x_optimal`` is not available

    """
    if getattr(problem, "x_optimal", None) is None:
        raise MetricUnavailableError("requires problem.x_optimal")

    x_opt = problem.x_optimal
    mean_x = x_mean(tuple(agents), iteration)
    optimal_cost = sum(a.cost.function(x_opt) for a in agents)  # type: ignore[arg-type, misc]
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


def x_error(agent: AgentMetricsView, problem: "BenchmarkProblem", iteration: int = -1) -> float:
    r"""
    Calculate x error at *iteration* (or at the agent's final x if *iteration* is -1).

    .. math::
        \|\mathbf{x}_k - \mathbf{x}^\star\|

    where :math:`\mathbf{x}_k` is the agent's local x at iteration k,
    and :math:`\mathbf{x}^\star` is the optimal x defined in the *problem*.

    Raises:
        MetricUnavailableError: if ``problem.x_optimal`` is not available

    """
    if getattr(problem, "x_optimal", None) is None:
        raise MetricUnavailableError("requires problem.x_optimal")

    agent_iteration = agent.x_history.max() if iteration == -1 else iteration
    x_at_iteration = iop.to_numpy(agent.x_history[agent_iteration])
    opt_x = iop.to_numpy(problem.x_optimal)  # type: ignore[arg-type]
    return float(la.norm(x_at_iteration - opt_x))


def accuracy(agents: Sequence[AgentMetricsView], problem: "BenchmarkProblem", iteration: int) -> list[float]:
    """
    Calculate the accuracy per agent.

    Accuracy is only applicable for problems using :class:`~decent_bench.costs.EmpiricalRiskCost`.

    Args:
        agents: sequence of agents to calculate accuracy for
        problem: benchmark problem containing test data
        iteration: iteration to calculate accuracy at, or -1 to use the agents' final x

    Returns:
        list of accuracies per agent at *iteration*

    Raises:
        MetricUnavailableError: if ``problem.test_data`` is missing, if any agent cost is not
            :class:`~decent_bench.costs.EmpiricalRiskCost`, or if the targets are not integer-valued

    """
    if getattr(problem, "test_data", None) is None:
        raise MetricUnavailableError("requires problem.test_data")
    if not all(isinstance(a.cost, costs.EmpiricalRiskCost) for a in agents):
        raise MetricUnavailableError("accuracy only applies if all agents have EmpiricalRiskCost")

    _, test_y = split_dataset(problem.test_data)  # type: ignore[arg-type]

    if test_y.dtype.kind not in {"i", "u"}:
        raise MetricUnavailableError(f"accuracy only applies for integer targets, dtype {test_y.dtype} found")

    ret: list[float] = []
    for agent in agents:
        agent_iteration = agent.x_history.max() if iteration == -1 else iteration
        preds = predict_agent(agent, agent_iteration, problem)
        ret.append(float(sk_metrics.accuracy_score(test_y, preds)))
    return ret


def mse(agents: Sequence[AgentMetricsView], problem: "BenchmarkProblem", iteration: int) -> list[float]:
    """
    Calculate the mean squared error (MSE) per agent.

    MSE is only applicable for problems using :class:`~decent_bench.costs.EmpiricalRiskCost`.

    Args:
        agents: sequence of agents to calculate MSE for
        problem: benchmark problem containing test data
        iteration: iteration to calculate MSE at, or -1 to use the agents' final x

    Returns:
        list of MSE per agent

    Raises:
        MetricUnavailableError: if ``problem.test_data`` is missing or if any agent cost is not
            :class:`~decent_bench.costs.EmpiricalRiskCost`

    """
    if getattr(problem, "test_data", None) is None:
        raise MetricUnavailableError("requires problem.test_data")
    if not all(isinstance(a.cost, costs.EmpiricalRiskCost) for a in agents):
        raise MetricUnavailableError("MSE only applies if all agents have EmpiricalRiskCost")

    ret: list[float] = []
    _, test_y = split_dataset(problem.test_data)  # type: ignore[arg-type]
    for agent in agents:
        agent_iteration = agent.x_history.max() if iteration == -1 else iteration
        preds = predict_agent(agent, agent_iteration, problem)
        ret.append(sk_metrics.mean_squared_error(test_y, preds))
    return ret


def precision(agents: Sequence[AgentMetricsView], problem: "BenchmarkProblem", iteration: int) -> list[float]:
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

    Raises:
        MetricUnavailableError: if ``problem.test_data`` is missing, if any agent cost is not
            :class:`~decent_bench.costs.EmpiricalRiskCost`, or if the targets are not integer-valued

    """
    if getattr(problem, "test_data", None) is None:
        raise MetricUnavailableError("requires problem.test_data")
    if not all(isinstance(a.cost, costs.EmpiricalRiskCost) for a in agents):
        raise MetricUnavailableError("precision only applies if all agents have EmpiricalRiskCost")

    _, test_y = split_dataset(problem.test_data)  # type: ignore[arg-type]

    if test_y.dtype.kind not in {"i", "u"}:
        raise MetricUnavailableError(f"precision only applies for integer targets, dtype {test_y.dtype} found")

    ret: list[float] = []
    for agent in agents:
        agent_iteration = agent.x_history.max() if iteration == -1 else iteration
        preds = predict_agent(agent, agent_iteration, problem)
        ret.append(float(sk_metrics.precision_score(test_y, preds, average="micro")))
    return ret


def recall(agents: Sequence[AgentMetricsView], problem: "BenchmarkProblem", iteration: int) -> list[float]:
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

    Raises:
        MetricUnavailableError: if ``problem.test_data`` is missing, if any agent cost is not
            :class:`~decent_bench.costs.EmpiricalRiskCost`, or if the targets are not integer-valued

    """
    if getattr(problem, "test_data", None) is None:
        raise MetricUnavailableError("requires problem.test_data")
    if not all(isinstance(a.cost, costs.EmpiricalRiskCost) for a in agents):
        raise MetricUnavailableError("recall only applies if all agents have EmpiricalRiskCost")

    _, test_y = split_dataset(problem.test_data)  # type: ignore[arg-type]

    if test_y.dtype.kind not in {"i", "u"}:
        raise MetricUnavailableError(f"recall only applies for integer targets, dtype {test_y.dtype} found")

    ret: list[float] = []
    for agent in agents:
        agent_iteration = agent.x_history.max() if iteration == -1 else iteration
        preds = predict_agent(agent, agent_iteration, problem)
        ret.append(float(sk_metrics.recall_score(test_y, preds, average="micro")))
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
def predict_agent(agent: AgentMetricsView, iteration: int, problem: "BenchmarkProblem") -> NDArray[float64]:
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

    """
    if not isinstance(agent.cost, costs.EmpiricalRiskCost):
        raise TypeError("Predictions can only be obtained for agents with EmpiricalRiskCost")

    if getattr(problem, "test_data", None) is None:
        raise ValueError("Test data is required to get predictions but is not provided in the problem")

    test_x, _ = split_dataset(problem.test_data)  # type: ignore[arg-type]

    return iop.to_numpy(agent.cost.predict(agent.x_history[iteration], list(test_x)))


def all_sorted_iterations(agents: Sequence[AgentMetricsView]) -> list[int]:
    """
    Get a sorted list of all iterations reached by any agent in *agents*.

    Args:
        agents: sequence of agents to get the iterations from

    Returns:
        sorted list of iterations reached by any agent

    """
    all_iters = set.union(*(set(a.x_history.keys()) for a in agents)) if agents else set()
    return sorted(all_iters)


def linear_convergence_rate(y: Sequence[float]) -> float:
    r"""
    Compute the linear (a.k.a. exponential or geometric) convergence rate from a given trajectory.

    Fits a piecewise linear model to the log10-scaled trajectory to identify
    the transitory phase and extract its slope. The convergence rate is then computed
    as :math:`10^{\text{slope}}`, giving the multiplicative factor by which the error
    decreases per iteration during the transitory phase. A convergence rate below :math:`1`
    indicates convergence, while above :math:`1` indicates divergence. The smaller the
    convergence rate, the faster the convergence.

    Args:
        y: sequence of error values from optimization trajectory (assumed to be positive)

    Returns:
        the convergence rate (multiplicative factor per iteration)

    Example:
        >>> print("Convergence rate of:")
        >>> for alg, results in metric_results.plot_results.items():
        >>>     for metric, stat_results in results.items():
        >>>         if type(metric) == metric_library.GradientNorm:
        >>>             print(f"\t- {alg.name}: {metric_utils.linear_convergence_rate(stat_results[1])}")

    """
    y_array: NDArray[float64] = np.asarray(y, dtype=float64)
    log_y: NDArray[float64] = np.log10(y_array)
    results = fit_elbow_curve(log_y)

    return float(10 ** results[1])


def fit_elbow_curve(
    y: NDArray[float64], max_trials: int = 10, tol: float = 1e-5, num_grid_points: int = 10
) -> tuple[float, float, int]:
    r"""
    Fit a piecewise linear "elbow curve" to data.

    Fits two connected line segments to the input data: one with a slope for
    the transitory phase and one horizontal for the steady-state phase. Formally, the elbow curve is defined as

    .. math::
            f(x) = \begin{cases}
                        s x + y_0 & \text{if } x \leq b \\
                        s b + y_0 & \text{if } x > b
                   \end{cases}

    where :math:`s` is the slope, :math:`y_0` is the intercept, :math:`b` the breakpoint.
    The parameters :math:`s`, :math:`y_0`, and :math:`b` are fitted to the input data using
    linear regression with an analytical solution (for efficiency), and grid search to find the optimal breakpoint.

    Args:
        y: 1D array of data points to fit
        max_trials: maximum number of refinement iterations for grid search
        tol: grid search stops when fit of residual is less than this value
        num_grid_points: number of candidate breakpoints to evaluate in each grid search iteration

    Returns:
        the *intercept*, *slope*, and *breakpoint* fitted to the data

    Note:
        A large value of *max_trials*, a small value of *tol*, a large value of *num_grid_points* will increase
        the accuracy of the fit, but will require longer computational time.

    Note:
        `numpy.nan`, `numpy.inf` or `-numpy.inf` values in *y* are disregarded during the fit. These values might
        occur in case of divergence (there is only a transient phase, with positive slope), and discarding them
        allows to still fit the slope.

    Raises:
        ValueError: if the input arguments are invalid

    """
    # validate arguments
    if len(y) < 2:
        raise ValueError("At least 2 data points are required to fit an elbow curve")

    if max_trials < 1:
        raise ValueError("max_trials must be at least 1")
    if tol < 0:
        raise ValueError("tol must be non-negative")
    if num_grid_points < 2:
        raise ValueError("num_grid_points must be at least 2")

    # discard inf and nan
    mask: NDArray[np.bool_] = np.isfinite(y)
    y = y[mask]
    m: int = int(y.size)  # num. datapoints

    # define search space for breakpoint b
    b_1: int = 1
    b_2: int = m - 1
    # initialize residual of current best fit
    best_res: float = float("inf")

    x_hat: NDArray[float64] = np.zeros((2, num_grid_points), dtype=float64)
    grid: NDArray[np.int_] = np.zeros(num_grid_points, dtype=int)

    for _ in range(max_trials):
        # build search grid for b
        grid = np.linspace(b_1, b_2, num=num_grid_points, dtype=int)

        x_hat = np.zeros((2, num_grid_points), dtype=float64)  # fitted parameters for each candidate breakpoint
        res: NDArray[float64] = np.zeros(num_grid_points, dtype=float64)  # residual for each candidate breakpoint

        # solve linear regression for points in grid
        for i, b in enumerate(grid):
            x_hat[:, i], res[i] = _fit_elbow_curve_given_breakpoint(y, b)
        best_idx: int = int(np.argmin(res))  # best candidate breakpoint

        # zoom in on region containing the best candidate
        b_1, b_2 = grid[max(0, best_idx - 1)], grid[min(num_grid_points - 1, best_idx + 1)]

        # stop if best residual did not improve significantly
        if best_res - res[best_idx] < tol:
            break

        best_res = res[best_idx]

    return float(x_hat[:, best_idx][1]), float(x_hat[:, best_idx][0]), int(grid[best_idx] - 1)


def _fit_elbow_curve_given_breakpoint(y: NDArray[float64], b: int) -> tuple[NDArray[float64], float]:
    """
    Perform least squares fit for piecewise linear model with fixed breakpoint.

    Fits a piecewise linear model where the first segment (0 to b) has a slope
    and intercept, and the second segment (b+1 to end) is horizontal at the
    same intercept value. Uses analytical solution for efficiency.

    Args:
        y: 1D array of data points to fit, as column of shape (m, 1)
        b: breakpoint index (0-based)

    Returns:
        the fitted parameters *x_hat* and *residual* of the fit

    Note:
        It is assumed that *y* does not contain `numpy.nan`, `numpy.inf` or `-numpy.inf`.

    """
    m = len(y)  # num. datapoints

    # build the regression matrix, assuming the breakpoint is b
    R: NDArray[float64] = np.hstack((  # noqa: N806
        np.vstack((
            np.arange(0, b + 1, 1).reshape((b + 1, 1)),
            b * np.ones((m - b - 1, 1)),
        )),
        np.ones((m, 1)),
    ))

    # analytical expression for inverse of R.T @ R
    S_00 = b * (b + 1) * (2 * b + 1) / 6.0 + (m - b - 1) * b**2  # noqa: N806
    S_01 = b * (b + 1) / 2.0 + (m - b - 1) * b  # noqa: N806
    S_11 = m  # noqa: N806

    S_inv: NDArray[float64] = np.array([[S_11, -S_01], [-S_01, S_00]]) / (S_00 * S_11 - S_01**2)  # noqa: N806

    # fit parameters and compute residual
    x_hat: NDArray[float64] = S_inv @ (R.T @ y)
    res = float(la.norm(R @ x_hat - y))

    # return fitted parameters and residual of fit
    return x_hat, res
