import operator
from collections.abc import Callable, Sequence
from functools import lru_cache, reduce
from typing import TYPE_CHECKING

import numpy as np
from decent_array import Array
from decent_array import interoperability as iop
from decent_array.types import dtypes
from numpy import float64
from numpy import linalg as la
from numpy.typing import NDArray
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
from rich.table import Column
from sklearn import metrics as sk_metrics

from decent_bench.costs import EmpiricalRiskCost
from decent_bench.metrics._metrics_view import AgentMetricsView
from decent_bench.networks import FedNetwork
from decent_bench.utils._logger import LOGGER

if TYPE_CHECKING:
    from decent_bench.benchmark import BenchmarkProblem


CACHE_MAX_SIZE = 50_000


class MetricProgressBar(Progress):
    """
    Progress bar for metric calculations.

    Make sure to set the field *status* in the task to show custom status messages.

    """

    def __init__(self) -> None:
        super().__init__(
            TextColumn(
                "[progress.description]{task.description}",
                table_column=Column(width=24, no_wrap=True),
            ),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(elapsed_when_finished=True),
            TextColumn("{task.fields[status]}"),
        )


def _clear_caches() -> None:
    """Clear module-level functools caches used by metric utilities."""
    x_mean.cache_clear()
    _predict.cache_clear()


@lru_cache(maxsize=CACHE_MAX_SIZE)
def x_mean(agents: tuple[AgentMetricsView, ...], iteration: int = -1) -> Array:
    """
    Calculate the mean x at *iteration* (or using the agents' final x if *iteration* is -1).

    Agents that did not reach *iteration* are disregarded.

    Raises:
        ValueError: if no agent reached *iteration*

    """
    all_x_at_iter = [a.x_history[iteration] for a in agents]

    if len(all_x_at_iter) == 0:
        raise ValueError(f"No agent reached iteration {iteration}")

    return reduce(operator.add, all_x_at_iter) / len(all_x_at_iter)


@lru_cache(maxsize=CACHE_MAX_SIZE)
def _predict(agent: AgentMetricsView, iteration: int, problem: "BenchmarkProblem") -> Array:
    """Get the predictions of *agent* at *iteration*. Cached since predictions may be expensive."""
    test_x = [x for x, _ in problem.test_data]  # type: ignore[union-attr]
    return agent.cost.predict(agent.x_history[iteration], test_x)  # type: ignore[no-any-return,attr-defined]


###### regression and classification metrics
def _test_targets(problem: "BenchmarkProblem") -> Array:
    """Get stacked test targets."""
    return iop.stack([y for _, y in problem.test_data])  # type: ignore[union-attr]


def _mse(agents: Sequence[AgentMetricsView], problem: "BenchmarkProblem", iteration: int) -> list[float]:
    """
    Compute the mean squared error (MSE) per agent.

    MSE is only applicable for problems using :class:`~decent_bench.costs.EmpiricalRiskCost`.

    Args:
        agents: sequence of agents to calculate MSE for
        problem: benchmark problem containing test data
        iteration: iteration to calculate MSE at, or -1 to use the agents' final x

    Returns:
        list of MSE per agent

    """
    ret: list[float] = []
    test_y = _test_targets(problem)

    for agent in agents:
        preds = _predict(agent, iteration, problem)

        if not iop.all(iop.isfinite(preds)):
            LOGGER.warning(
                "Predictions contain NaN or Inf values, which are not valid for MSE calculation, returning NaN"
            )
            return [np.nan for _ in agents]

        error = preds - test_y
        ret.append(float(iop.mean(error * error)))

    return ret


def _accuracy(agents: Sequence[AgentMetricsView], problem: "BenchmarkProblem", iteration: int) -> list[float]:
    """
    Compute the accuracy per agent.

    Accuracy is only applicable for problems using :class:`~decent_bench.costs.EmpiricalRiskCost`.

    Args:
        agents: sequence of agents to calculate accuracy for
        problem: benchmark problem containing test data
        iteration: iteration to calculate accuracy at, or -1 to use the agents' final x

    Returns:
        list of accuracies per agent at *iteration*

    """
    test_y = _test_targets(problem)
    ret: list[float] = []

    for agent in agents:
        preds = _predict(agent, iteration, problem)

        if not iop.all(iop.isfinite(preds)):
            LOGGER.warning(
                "Predictions contain NaN or Inf values, which are not valid for accuracy calculation, returning NaN"
            )
            return [np.nan for _ in agents]

        ret.append(float(iop.mean(preds == test_y)))

    return ret


def _precision(agents: Sequence[AgentMetricsView], problem: "BenchmarkProblem", iteration: int) -> list[float]:
    """
    Compute the precision per agent.

    Precision is only applicable for problems using :class:`~decent_bench.costs.EmpiricalRiskCost`.
    Calculated using :func:`sklearn.metrics.precision_score` with micro averaging.

    Args:
        agents: sequence of agents to calculate precision for
        problem: benchmark problem containing test data
        iteration: iteration to calculate precision at, or -1 to use the agents' final x

    Returns:
        list of precision per agent at *iteration*

    """
    test_y = iop.to_numpy(_test_targets(problem))
    ret: list[float] = []

    for agent in agents:
        preds = _predict(agent, iteration, problem)

        if not iop.all(iop.isfinite(preds)):
            LOGGER.warning(
                "Predictions contain NaN or Inf values, which are not valid for precision calculation, returning NaN"
            )
            return [np.nan for _ in agents]

        ret.append(float(sk_metrics.precision_score(test_y, iop.to_numpy(preds), average="micro")))

    return ret


def _recall(agents: Sequence[AgentMetricsView], problem: "BenchmarkProblem", iteration: int) -> list[float]:
    """
    Compute the recall per agent.

    Recall is only applicable for problems using :class:`~decent_bench.costs.EmpiricalRiskCost`.
    Calculated using :func:`sklearn.metrics.recall_score` with micro averaging.

    Args:
        agents: sequence of agents to calculate recall for
        problem: benchmark problem containing test data
        iteration: iteration to calculate recall at, or -1 to use the agents' final x

    Returns:
        list of recall per agent at *iteration*

    """
    test_y = iop.to_numpy(_test_targets(problem))
    ret: list[float] = []

    for agent in agents:
        preds = _predict(agent, iteration, problem)

        if not iop.all(iop.isfinite(preds)):
            LOGGER.warning(
                "Predictions contain NaN or Inf values, which are not valid for recall calculation, returning NaN"
            )
            return [np.nan for _ in agents]

        ret.append(float(sk_metrics.recall_score(test_y, iop.to_numpy(preds), average="micro")))

    return ret


###### availability tests for metrics
def _requires_x_optimal(problem: "BenchmarkProblem") -> str | None:
    if getattr(problem, "x_optimal", None) is None:
        return "requires problem.x_optimal"
    return None


def _requires_test_data(problem: "BenchmarkProblem") -> str | None:
    if getattr(problem, "test_data", None) is None:
        return "requires problem.test_data"
    return None


def _requires_empirical_cost(problem: "BenchmarkProblem") -> str | None:
    if not all(isinstance(a.cost, EmpiricalRiskCost) for a in problem.network.agents()):
        return "requires all agents with EmpiricalRiskCost"
    return None


def _requires_integer_targets(problem: "BenchmarkProblem") -> str | None:
    reason = _requires_test_data(problem)
    if reason is not None:
        return reason

    int_dtypes = dtypes(kind=("signed integer", "unsigned integer"))
    for d in problem.test_data:  # type: ignore[union-attr]
        if d[1].dtype not in int_dtypes:
            return f"requires integer targets, dtype {d[1].dtype} found"
    return None


def _requires_fednetwork(problem: "BenchmarkProblem") -> str | None:
    if not isinstance(problem.network, FedNetwork):
        return "requires FedNetwork"
    return None


def _check_availability(
    conditions: tuple[Callable[["BenchmarkProblem"], str | None], ...], problem: "BenchmarkProblem"
) -> str | None:
    for c in conditions:
        reason = c(problem)
        if reason is not None:
            return reason
    return None


###### iterations utils
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


def _observed_rounds(agents: Sequence[AgentMetricsView]) -> int:
    """
    Get the number of completed rounds observed in the agents' histories.

    The initial snapshot is stored at iteration 0, so the maximum recorded iteration corresponds to the number of
    completed algorithm rounds when the final snapshot is present.
    """
    if not agents:
        return 0
    return max(agent.x_history.max() for agent in agents)


###### convergence rate
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
        >>> xerror = metrics_results.plot_results[metrics_results.plot_results["metric"] == "x error"]
        >>> for alg in metrics_results.algorithms:
                xerror_over_time = xerror[xerror["algorithm"] == alg]["mean"]
        >>>     print(f"Rate for {alg} = {linear_convergence_rate(xerror[xerror["algorithm"] == alg]["mean"])}")

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
    R: NDArray[float64] = np.hstack(  # noqa: N806
        (
            np.vstack(
                (
                    np.arange(0, b + 1, 1).reshape((b + 1, 1)),
                    b * np.ones((m - b - 1, 1)),
                )
            ),
            np.ones((m, 1)),
        )
    )

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
