import numpy as np
from numpy import float64
from numpy import linalg as la
from numpy.typing import NDArray

from decent_bench.library.core.cost_functions import CostFunction


def gradient_descent(
    cost_function: CostFunction,
    x0: NDArray[float64],
    *,
    step_size: float,
    max_iter: int,
    stop_tol: float | None,
    max_tol: float | None,
) -> NDArray[float64]:
    """
    Find the x that minimizes the cost function using gradient descent.

    Args:
        cost_function: cost function to minimize
        x0: initial guess
        step_size: scaling factor for each update
        max_iter: maximum number of iterations to run
        stop_tol: early stopping criteria - stop if ``norm(x_new - x) <= stop_tol``
        max_tol: maximum tolerated ``norm(x_new - x)`` at the end

    Raises:
        RuntimeError: if ``norm(x_new - x) > max_tol`` at the end

    Returns:
        x that minimizes the cost function.

    """
    delta = np.inf
    x = x0
    for _ in range(max_iter):
        x_new = x - step_size * cost_function.gradient(x)
        delta = float(la.norm(x_new - x))
        x = x_new
        if stop_tol is not None and delta <= stop_tol:
            break
    if max_tol is not None and delta > max_tol:
        raise RuntimeError(
            f"Gradient descent failed to reach convergence within {max_iter} iterations with step size {step_size}."
            f"Max delta acceptable: {max_tol}."
            f"Actual delta: {delta}."
        )
    return x


def accelerated_gradient_descent(
    cost_function: CostFunction,
    x0: NDArray[float64] | None,
    *,
    max_iter: int,
    stop_tol: float | None,
    max_tol: float | None,
) -> NDArray[float64]:
    r"""
    Find the x that minimizes the cost function using accelerated gradient descent.

    Args:
        cost_function: cost function to minimize
        x0: initial guess, defaults to ``np.zeros()`` if ``None`` is provided
        max_iter: maximum number of iterations to run
        stop_tol: early stopping criteria - stop if ``norm(x_new - x) <= stop_tol``
        max_tol: maximum tolerated ``norm(x_new - x)`` at the end

    Raises:
        RuntimeError: if ``norm(x_new - x) > max_tol`` at the end
        ValueError: if ``cost_function.m_smooth < 0``, ``cost_function.m_cvx < 0``, or cost function is affine

    Returns:
        x that minimizes the cost function.

    """
    if x0 is not None and x0.shape != cost_function.domain_shape:
        raise ValueError("x0 and cost function domain must have same shape")
    if cost_function.m_smooth == 0:
        raise ValueError("Function must not be affine")
    if cost_function.m_smooth < 0:
        raise ValueError("m_smooth must not be negative")
    if cost_function.m_cvx < 0:
        raise ValueError("m_cvx must not be negative")
    if cost_function.m_smooth == np.inf:
        raise NotImplementedError("Support for non-L-smoothness is not implemented yet")
    if np.isnan(cost_function.m_smooth):
        raise NotImplementedError("Support for non-global differentiability is not implemented yet")
    if np.isnan(cost_function.m_cvx):
        raise NotImplementedError("Support for non-convexity is not implemented yet")
    x0 = x0 if x0 is not None else np.zeros(cost_function.domain_shape)
    x = x0
    y = x0
    c = (np.sqrt(cost_function.m_smooth) - np.sqrt(cost_function.m_cvx)) / (
        np.sqrt(cost_function.m_smooth) + np.sqrt(cost_function.m_cvx)
    )
    delta = np.inf
    for k in range(1, max_iter + 1):
        x_new = y - cost_function.gradient(y) / cost_function.m_smooth
        delta = float(la.norm(x_new - x))
        beta = c if cost_function.m_cvx > 0 else (k - 1) / (k + 2)
        y_new = x_new + beta * (x_new - x)
        x, y = x_new, y_new
        if stop_tol is not None and delta <= stop_tol:
            break
    if max_tol is not None and delta > max_tol:
        raise RuntimeError(
            f"Accelerated gradient descent failed to reach convergence within {max_iter} iterations."
            f"Max delta acceptable: {max_tol}."
            f"Actual delta: {delta}."
        )
    return x
