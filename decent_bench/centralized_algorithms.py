from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, final

import numpy as np
from rich.progress import track

import decent_bench.utils.interoperability as iop
from decent_bench.utils import logger
from decent_bench.utils.array import Array
from decent_bench.utils.logger import LOGGER

if TYPE_CHECKING:
    from decent_bench.costs import Cost


def solve(cost: "Cost", max_iter: int = 100, stop_tol: float | None = None, max_tol: float | None = None) -> Array:
    """
    Minimize a cost function using a suitable solver.

    Applies :func:`~numpy.linalg.solve` to quadratic costs, accelerated gradient descent to smooth and (strongly) convex
    costs, (sub)gradient descent to any other cost.

    Args:
        cost: cost function to minimize.
        max_iter: maximum number of iterations to run. Defaults to 100.
        stop_tol: optional early stopping tolerance; stops if ``||x_new - x_old||^2`` drops below this value.
        max_tol: optional final tolerance; RuntimeError is raised if ``||x_new - x_old||^2`` exceeds this value
                 after max_iter iterations.

    Returns:
        Approximate minimizer or stationary point.

    Raises:
        ValueError: when the cost has m_smooth = 0.

    """
    if not LOGGER.handlers:
        logger.start_logger()
    LOGGER.info("Finding the optimal solution to the problem ...")

    stop_criteria = f"Stopping after {max_iter} iterations"
    if stop_tol is not None:
        stop_criteria += f" or when ||x_new - x_old||^2 <= {stop_tol}"
    stop_criteria += "."
    if max_tol is not None:
        stop_criteria += f" Will raise if ||x_new - x_old||^2 > {max_tol} at the end."

    # quadratic
    from decent_bench.costs import QuadraticCost  # noqa: PLC0415

    if isinstance(cost, QuadraticCost):
        x_optimal = Array(np.linalg.solve(cost.A, -cost.b))
    # exclude costs with m_smooth = 0
    elif np.isfinite(cost.m_smooth) and np.isfinite(cost.m_cvx) and cost.m_smooth == 0:
        raise ValueError("Costs with m_smooth = 0 are not supported.")
    # smooth and convex/strongly convex
    elif np.isfinite(cost.m_smooth) and np.isfinite(cost.m_cvx) and cost.m_smooth > 0:
        LOGGER.info(f"{stop_criteria}")
        x_optimal = AcceleratedGradientDescent(cost).run(max_iter=max_iter, stop_tol=stop_tol, max_tol=max_tol)
    # non-smooth or non-convex
    else:
        LOGGER.info(f"{stop_criteria}")
        x_optimal = GradientDescent(cost).run(max_iter=max_iter, stop_tol=stop_tol, max_tol=max_tol)

    LOGGER.info("... done!")

    return x_optimal


class Solver(ABC):
    """
    Base class for centralized solvers.

    Initializes iterate (x) and previous iterate (x_old), validates domain shape,
    and stores hyperparameters.
    Subclasses must implement the step method to define one iteration of their algorithm.
    """

    def __init__(self, cost: "Cost", x0: Array | None = None):
        if x0 is None:
            x0 = iop.zeros(shape=cost.shape, framework=cost.framework, device=cost.device)
        if iop.shape(x0) != cost.shape:
            raise ValueError("x0 and cost function domain must have same shape")
        self.x = x0
        self.x_old = iop.copy(self.x)

        self.cost = cost

    @abstractmethod
    def step(self, iteration: int) -> None:
        """
        Perform one iteration of the solver.

        Subclasses must update self.x exactly once per step.
        Use the iteration counter for algorithms with iteration-dependent parameters (e.g., step schedules).

        Args:
            iteration: current iteration number.

        """

    @final
    def run(
        self,
        max_iter: int = 100,
        stop_tol: float | None = None,
        max_tol: float | None = None,
        check_frequency: float = 0.01,
        show_progress: bool = True,
    ) -> Array:
        """
        Run the solver.

        Executes :meth:`step` for up to max_iter iterations. Stops early if the squared norm of the iterate change
        drops below stop_tol. After completion, verifies that the final iterate change is at most max_tol.

        Args:
            max_iter: maximum number of iterations; must be positive. Defaults to 100.
            stop_tol: optional early stopping tolerance; stops if ``||x_new - x_old||^2 <= stop_tol``.
                      Must be positive if provided.
            max_tol: optional final tolerance; raises RuntimeError if ``||x_new - x_old||^2 > max_tol`` after
                      max_iter iterations. Must be positive if provided.
            check_frequency: float in (0, 1] defining how often the early stopping condition should be checked.
                      A smaller value means that the stopping condition is checked more often.
                      This applies only if ``stop_tol`` is not None.
            show_progress: whether to display a progress bar during iteration. Defaults to True.

        Returns:
            Final iterate x as an Array.

        Raises:
            ValueError: if max_iter < 1, or if stop_tol or max_tol are provided and non-positive.
            RuntimeError: if max_tol is provided and the final iterate change exceeds max_tol.

        Warning:
            Do not override this method. Instead, override :meth:`step` to define solver behavior.

        """
        if max_iter < 1:
            raise ValueError("`max_iter` must be positive")
        if stop_tol is not None and stop_tol <= 0:
            raise ValueError("`stop_tol` must be positive or None")
        if max_tol is not None and max_tol <= 0:
            raise ValueError("`max_tol` must be positive or None")
        if check_frequency <= 0 or check_frequency > 1:
            raise ValueError("`check_frequency` must be a float in (0, 1]")
        check_every = max(1, int(check_frequency * max_iter))

        for k in track(
            range(max_iter),
            description="Solving...",
            disable=not show_progress,
            update_period=0.0,
        ):
            self.x_old = iop.copy(self.x)
            self.step(k)
            if stop_tol is not None and k % check_every == 0:
                d = self.x - self.x_old
                delta = float(iop.transpose(d) @ d)
                if delta <= stop_tol:
                    break

        if max_tol is not None:
            if stop_tol is None or k % check_every != 0:
                d = self.x - self.x_old
                delta = float(iop.transpose(d) @ d)
            if delta > max_tol:
                raise RuntimeError(
                    f"Solver failed to converge within {max_iter} iterations: delta {delta} > max delta {max_tol}."
                )

        return self.x


class GradientDescent(Solver):
    r"""
    Gradient descent solver.

    If step_size is not provided, defaults to:
    - Non-smooth or non-convex: :math:`1/\sqrt{k+1}`
    - Strongly convex: :math:`2/(L+mu)`
    - Convex: step_size = 1/m_smooth
    """

    def __init__(self, cost: "Cost", step_size: float | Callable[[int], float] | None = None, x0: Array | None = None):
        if callable(step_size):
            step_size_k: Callable[[int], float] = step_size
        elif isinstance(step_size, float):
            step_size_k = lambda _: float(step_size)  # noqa: E731
        elif np.isnan(cost.m_smooth) or np.isinf(cost.m_smooth) or np.isnan(cost.m_cvx):  # non-smooth or non-convex
            step_size_k = lambda k: float(1 / np.sqrt(k + 1))  # noqa: E731
        elif cost.m_cvx > 0:  # strongly convex
            step_size_k = lambda _: 2 / (cost.m_smooth + cost.m_cvx)  # noqa: E731
        else:  # convex
            step_size_k = lambda _: 1 / cost.m_smooth  # noqa: E731

        super().__init__(cost, x0)
        self.step_size = step_size_k

    def step(self, iteration: int) -> None:
        """Perform one iteration of the solver."""
        self.x -= self.step_size(iteration) * self.cost.gradient(self.x)


class AcceleratedGradientDescent(Solver):
    r"""
    Accelerated gradient descent (Nesterov momentum) solver.

    If step_size is not provided, defaults to: :math:`1/L`.

    If momentum is not provided, defaults to:
    - Strongly convex: :math:`(\sqrt(L)-\sqrt(mu)) / (\sqrt(L)+\sqrt(mu))`
    - Otherwise: :math:`k / (k+3)`
    """

    def __init__(
        self,
        cost: "Cost",
        step_size: float | None = None,
        momentum: float | Callable[[int], float] | None = None,
        x0: Array | None = None,
    ):
        step_size = float(step_size) if isinstance(step_size, float) else 1 / cost.m_smooth

        if callable(momentum):
            momentum_k: Callable[[int], float] = momentum
        elif isinstance(momentum, float):
            momentum_k = lambda _: float(momentum)  # noqa: E731
        elif cost.m_cvx > 0:  # strongly convex
            momentum_k = lambda _: float(  # noqa: E731
                (np.sqrt(cost.m_smooth) - np.sqrt(cost.m_cvx)) / (np.sqrt(cost.m_smooth) + np.sqrt(cost.m_cvx))
            )
        else:
            momentum_k = lambda k: k / (k + 3)  # noqa: E731

        super().__init__(cost, x0)
        self.step_size = step_size
        self.momentum = momentum_k
        self.y = iop.copy(self.x)

    def step(self, iteration: int) -> None:
        """Perform one iteration of the solver."""
        self.x = self.y - self.step_size * self.cost.gradient(self.y)
        self.y = self.x + self.momentum(iteration) * (self.x - self.x_old)


def proximal_solver(cost: "Cost", y: Array, rho: float, max_iter: int = 100) -> Array:
    """
    Approximate the cost's proximal at y using accelerated gradient descent.

    This is an approximate solution to the proximal operator defined as:

    .. include:: snippets/proximal_operator.rst

    The cost must be differentiable, L-smooth, and convex.

    Args:
        cost: cost function to compute the proximal of.
        y: point at which to evaluate the proximal.
        rho: penalty parameter.
        max_iter: maximum number of iterations of the solver.

    Returns:
        Approximate proximal at `y`.

    Raises:
        ValueError: if cost's domain and `y` do not have the same shape, or if `rho` is not positive.
        NotImplementedError: if the cost is not differentiable, L-smooth, and convex.

    """
    if cost.shape != iop.shape(y):
        raise ValueError("Cost function domain and y need to have the same shape")
    if rho <= 0:
        raise ValueError("Penalty term `rho` must be greater than 0")

    from decent_bench.costs import QuadraticCost  # noqa: PLC0415

    proximal_cost = QuadraticCost(A=iop.eye_like(y) / rho, b=-y / rho) + cost
    if proximal_cost.m_smooth == np.inf or np.isnan(proximal_cost.m_smooth) or np.isnan(proximal_cost.m_cvx):
        raise NotImplementedError("Proximal solver requires the cost to be differentiable, L-smooth, and convex.")
    return AcceleratedGradientDescent(proximal_cost, x0=y).run(
        max_iter=max_iter, stop_tol=1e-10, max_tol=None, show_progress=False
    )
