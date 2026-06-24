Cost Function
-------------

You can use built-in costs or implement a custom cost by subclassing :class:`~decent_bench.costs.Cost`.

Cost composition supports arithmetic patterns such as:

- ``cost + regularizer``
- ``scalar * cost``
- ``sum(costs)``

For empirical objectives, compose carefully so batch-aware behavior is preserved.



Available costs
---------------

Regression
~~~~~~~~~~
:tagged:`cost, regression`

Classification
~~~~~~~~~~~~~~
:tagged:`cost, regression`


PyTorchCost regularization
~~~~~~~~~~~~~~~~~~~~~~~~~~
When combining :class:`~decent_bench.costs.PyTorchCost` with one of the
built-in regularizers, instantiate the regularizer with the same framework
and device as the empirical cost:

.. code-block:: python

    from decent_bench.costs import L2RegularizerCost
    from decent_bench.utils.types import SupportedFrameworks

    reg = L2RegularizerCost(
        shape=cost.shape,
        framework=SupportedFrameworks.PYTORCH,
        device=cost.device,
    )
    objective = cost + reg

This preserves compatibility with the PyTorch empirical objective and keeps
the resulting objective in the empirical, batch-compatible abstraction.
It is convenient for composition, but it is not necessarily the most
efficient option compared with native framework-specific regularization.





Create problems from scratch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Create a custom benchmark problem with your own dataset, cost function, and communication schemes by implementing the
corresponding abstracts.

.. code-block:: python

    import networkx as nx

    from decent_bench import benchmark
    from decent_bench import centralized_algorithms as ca
    from decent_bench.benchmark import BenchmarkProblem
    from decent_bench.costs import Cost
    from decent_bench.datasets import DatasetHandler
    from decent_bench.distributed_algorithms import DGD, SimpleGT
    from decent_bench.schemes import AgentActivationScheme, CompressionScheme, DropScheme, NoiseScheme

    class MyDataset(DatasetHandler): ... # Optional but convienient to manage partitions

    class MyCost(Cost): ...

    class MyAgentActivationScheme(AgentActivationScheme): ...

    class MyCompressionScheme(CompressionScheme): ...

    class MyNoiseScheme(NoiseScheme): ...

    class MyDropScheme(DropScheme): ...

    n_agents = 100

    costs = [MyCost(*p) for p in MyDataset().training_partitions()]

    sum_cost = sum(costs[1:], start=costs[0])
    x_optimal = ca.accelerated_gradient_descent(
        sum_cost, x0=None, max_iter=50000, stop_tol=1e-100, max_tol=1e-16
    )

    problem = BenchmarkProblem(
        network_structure=nx.random_regular_graph(3, n_agents, seed=0),
        costs=costs,
        x_optimal=x_optimal,
        agent_activations=[MyAgentActivationScheme()] * n_agents,
        message_compression=MyCompressionScheme(),
        message_noise=MyNoiseScheme(),
        message_drop=MyDropScheme(),
    )

    if __name__ == "__main__":
        benchmark_result = benchmark.benchmark(
            algorithms=[DGD(iterations=1000, step_size=0.01), SimpleGT(iterations=1000, step_size=0.01)],
            benchmark_problem=problem,
        )
        metrics_result = benchmark.compute_metrics(benchmark_result)
        benchmark.display_metrics(metrics_result)







Cost Functions
--------------
Create new cost functions by subclassing :class:`~decent_bench.costs.Cost` and using interoperability decorators to keep
your implementation framework-agnostic. The decorators automatically wrap inputs/outputs as `Array` and ensure
compatibility with the selected framework and device of your custom cost.
Composition preserves specialized structure when possible, and otherwise falls back to generic wrappers.

Basic Operations
~~~~~~~~~~~~~~~~
Supported operations for cost objects:

- Addition: ``cost_a + cost_b``
- Subtraction: ``cost_a - cost_b``
- Negation: ``-cost``
- Scalar multiplication: ``scalar * cost`` or ``cost * scalar``
- Scalar division: ``cost / scalar``
- Summation: ``sum(costs)`` (uses ``__radd__``)

Composition Rules
~~~~~~~~~~~~~~~~~
Cost arithmetic preserves specialized structure for the most common composition patterns and falls back to generic
wrappers otherwise. When a composition falls back to :class:`~decent_bench.costs.SumCost` or
:class:`~decent_bench.costs.ScaledCost`, the result only guarantees the base :class:`~decent_bench.costs.Cost`
interface.

- ``regularizer_a + regularizer_b``, ``regularizer_a - regularizer_b``, ``scalar * regularizer``,
  ``regularizer / scalar``, and ``-regularizer`` preserve a regularizer-aware cost.
- ``scalar * empirical_cost`` preserves the empirical-risk interface through an internal empirical scaling wrapper.
- ``empirical_cost + regularizer`` and ``empirical_cost - regularizer`` preserve the empirical-risk interface through
  :class:`~decent_bench.costs.EmpiricalRegularizedCost`.
- Unsupported combinations still fall back to the generic wrappers
  :class:`~decent_bench.costs.SumCost` and :class:`~decent_bench.costs.ScaledCost`.

Regularization
~~~~~~~~~~~~~~
Regularized objectives can be built by composing cost functions with arithmetic. Decent-Bench provides the following
built-in regularizers:

The canonical regularization pattern is ``objective = cost + lambda_ * regularizer``.

- :class:`~decent_bench.costs.L1RegularizerCost` for :math:`\|x\|_1`
- :class:`~decent_bench.costs.L2RegularizerCost` for :math:`\frac{1}{2}\|x\|_2^2`
- :class:`~decent_bench.costs.FractionalQuadraticRegularizerCost` for
  :math:`\sum_i \frac{x_i^2}{1 + x_i^2}` (nonconvex)

All built-in regularizers accept and ignore empirical-risk-specific kwargs (for example ``indices="batch"``), so
batching continues to work when you compose them with empirical risk costs.

Empirical Risk Composition
~~~~~~~~~~~~~~~~~~~~~~~~~~
Supported empirical-risk compositions preserve empirical-risk-specific behavior such as ``predict``, ``dataset``,
``n_samples``, ``batch_size``, and batch helpers.

In particular, ``objective = cost + regularizer`` returns an
:class:`~decent_bench.costs.EmpiricalRegularizedCost`, which combines the empirical and regularizer contributions in
``function``, ``gradient``, and ``hessian`` while preserving the empirical interface of the base loss.

When using :class:`~decent_bench.costs.PyTorchCost`, prefer PyTorch's built-in loss regularizers for better
efficiency; iop regularizers remain available for cross-framework compatibility.

Examples
~~~~~~~~
.. code-block:: python

    from decent_bench.costs import (
        LogisticRegressionCost,
        L1RegularizerCost,
        L2RegularizerCost,
        FractionalQuadraticRegularizerCost,
    )

    cost = LogisticRegressionCost(dataset=dataset, batch_size="all")

    lam = 0.1
    eps = 0.01
    l1 = lam * L1RegularizerCost(shape=cost.shape)
    l2 = lam * L2RegularizerCost(shape=cost.shape)
    fq = eps * FractionalQuadraticRegularizerCost(shape=cost.shape)

    regularizer = l1 + l2
    objective = cost + regularizer
    nonconvex_objective = objective + fq

.. code-block:: python

    lambda_ = 0.05
    objective = cost + lambda_ * L2RegularizerCost(shape=cost.shape)
    value = objective.function(x, indices="all")
    gradient = objective.gradient(x, indices="all")

.. code-block:: python

    import numpy as np
    from decent_bench.costs import QuadraticCost

    arbitrary = QuadraticCost(np.eye(cost.shape[0]), np.zeros(cost.shape[0]))
    generic = objective + arbitrary  # returns SumCost

Important Semantics
~~~~~~~~~~~~~~~~~~~
Reduction Semantics
^^^^^^^^^^^^^^^^^^^
:class:`~decent_bench.costs.EmpiricalRegularizedCost.gradient` uses broadcast semantics when ``reduction=None``: the
empirical term returns one gradient per selected sample, and the regularizer gradient is added to each row. Averaging
over the sample dimension recovers the same composite gradient returned by ``reduction="mean"``.

Proximal Semantics
^^^^^^^^^^^^^^^^^^
.. warning::

    Proximal support is intentionally conservative. Positive scalar scaling preserves proximal support, and a single
    positively scaled regularizer term preserves the underlying regularizer proximal. Multi-term regularizer
    composites and :class:`~decent_bench.costs.EmpiricalRegularizedCost` do not define a generic proximal. Use a
    specialized proximal if one exists, or rely on :func:`decent_bench.centralized_algorithms.proximal_solver` when
    its assumptions are satisfied.

.. warning::

    :class:`~decent_bench.costs.SumCost.proximal` computes the proximal of the full summed objective through
    :func:`decent_bench.centralized_algorithms.proximal_solver`, which uses accelerated gradient descent. This
    requires the summed objective to satisfy that backend's assumptions, in particular differentiability, global
    smoothness, and convexity.

Copy Semantics
^^^^^^^^^^^^^^
.. warning::

    Composition wrappers keep references to the underlying cost objects; they do not make implicit copies. Mutating a
    reused cost after composition therefore affects all wrappers that reference it. Agent-installed call-counting
    hooks on reused cost objects are also shared. Use ``copy.deepcopy`` when independent composed objects or
    independent counting behavior are required.

.. code-block:: python

    import copy

    shared = cost + cost
    independent = copy.deepcopy(shared)

Custom Cost Example
~~~~~~~~~~~~~~~~~~~
.. code-block:: python

    from numpy import float64
    from numpy.typing import NDArray

    import decent_bench.utils.interoperability as iop
    from decent_bench.costs import Cost
    from decent_bench.utils.types import SupportedFrameworks, SupportedDevices

    class MyCost(Cost):
        def __init__(self, A: Array, b: Array):
            # Convert any external arrays to Array using the chosen framework/device
            self.A: NDArray[float64] = iop.to_numpy(A)
            self.b: NDArray[float64] = iop.to_numpy(b)

        @property
        def shape(self) -> tuple[int, ...]:
            # Domain shape (e.g., dimension of x)
            return (self.A.shape[1],)

        @property
        def framework(self) -> str:
            return SupportedFrameworks.NUMPY

        @property
        def device(self) -> str | None:
            return SupportedDevices.CPU

        @property
        def m_smooth(self) -> float:
            # Provide a meaningful smoothness constant if available
            return 1.0

        @property
        def m_cvx(self) -> float:
            # Provide convexity constant (0 if non-strongly convex)
            return 0.0

        @iop.autodecorate_cost_method(Cost.function)
        def function(self, x: NDArray[float64]) -> float:
            # Return a scalar (float) or Array scalar compatible with the framework
            r = self.A @ x - self.b
            return 0.5 * float(iop.dot(r, r))

        @iop.autodecorate_cost_method(Cost.gradient)
        def gradient(self, x: NDArray[float64]) -> NDArray[float64]:
            # Return an Array with same shape as x
            return self.A.T @ (self.A @ x - self.b)

        @iop.autodecorate_cost_method(Cost.hessian)
        def hessian(self, x: NDArray[float64]) -> NDArray[float64]:
            # Optional: return an Array representing the Hessian
            return self.A.T @ self.A

        @iop.autodecorate_cost_method(Cost.proximal)
        def proximal(self, x: NDArray[float64], rho: float) -> NDArray[float64]:
            # Optional: provide a closed-form proximal if available
            # Otherwise you can rely on `centralized_algorithms.proximal_solver`.
            return x  # identity as a placeholder

        # No __add__ implementation is required unless you want to preserve
        # a more specialized structure than the generic Cost fallback.
