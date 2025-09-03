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
    x0: NDArray[float64],
    *,
    m_smooth: float,
    m_cvx: float,
    max_iter: int,
    stop_tol: float | None,
    max_tol: float | None,
) -> NDArray[float64]:
    r"""
    Find the x that minimizes the cost function using accelerated gradient descent.

    Args:
        cost_function: cost function to minimize
        x0: initial guess
        m_smooth: Lipschitz constant of the cost function's gradient, i.e. the smallest value such that
            :math:`||\nabla f(\mathbf{x_1}) - \nabla f(\mathbf{x_2})|| \leq \text{m_smooth}
            \cdot \|\mathbf{x_1} - \mathbf{x_2}\|`
            for all :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`
        m_cvx: strong convexity constant of the cost function, i.e. the largest value such that
            :math:`f(\mathbf{x_1})
            \geq f(\mathbf{x_2})
            + \nabla f(\mathbf{x_2})^T (\mathbf{x_1} - \mathbf{x_2})
            + \frac{\text{m_cvx}}{2} \|\mathbf{x_1} - \mathbf{x_2}\|^2`
            for all :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`.
            If zero, the method assumes convexity (not strong convexity) and uses a time-varying momentum schedule
            instead of a constant one.
        max_iter: maximum number of iterations to run
        stop_tol: early stopping criteria - stop if ``norm(x_new - x) <= stop_tol``
        max_tol: maximum tolerated ``norm(x_new - x)`` at the end

    Raises:
        RuntimeError: if ``norm(x_new - x) > max_tol`` at the end
        ValueError: if *m_smooth* or *m_cvx* is negative

    Returns:
        x that minimizes the cost function.

    """
    if m_cvx < 0:
        raise ValueError("Strong convexity constant must be non-negative")
    if m_smooth < 0:
        raise ValueError("Lipschitz constant must be non-negative")
    if m_smooth == np.inf:
        raise NotImplementedError(
            "Support for unknown Lipschitz constant is not implemented yet, please use non-accelerated gradient descent"
        )
    x = x0
    y = x0
    c = (np.sqrt(m_smooth) - np.sqrt(m_cvx)) / (np.sqrt(m_smooth) + np.sqrt(m_cvx))
    delta = np.inf
    for k in range(1, max_iter + 1):
        x_new = y - cost_function.gradient(y) / m_smooth
        delta = float(la.norm(x_new - x))
        beta = c if m_cvx > 0 else (k - 1) / (k + 2)
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
