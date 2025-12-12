User Guide
==========
This user guide shows you different examples of how to use decent-bench.


Installation
------------
Requires `Python 3.13+ <https://www.python.org/downloads/>`_

.. code-block:: bash

    pip install decent-bench


Sunny case
----------
Benchmark algorithms on a regression problem without any communication constraints, using only default settings.

.. code-block:: python

    from decent_bench import benchmark, benchmark_problem
    from decent_bench.costs import LinearRegressionCost
    from decent_bench.distributed_algorithms import ADMM, DGD, ED

    if __name__ == "__main__":
        benchmark.benchmark(
            algorithms=[
                DGD(iterations=1000, step_size=0.001),
                ED(iterations=1000, step_size=0.001),
                ADMM(iterations=1000, rho=10, alpha=0.3),
            ],
            benchmark_problem=benchmark_problem.create_regression_problem(LinearRegressionCost),
        )


Benchmark executions will have outputs like these:

.. list-table::

   * - .. image:: _static/table.png
          :align: center
     - .. image:: _static/plot.png
          :align: center


Execution settings
------------------
Configure settings for metrics, trials, statistical confidence level, logging, and multiprocessing.

.. code-block:: python

    from logging import DEBUG

    from decent_bench import benchmark, benchmark_problem
    from decent_bench.costs import LinearRegressionCost
    from decent_bench.distributed_algorithms import ADMM, DGD
    from decent_bench.metrics.plot_metrics import RegretPerIteration
    from decent_bench.metrics.table_metrics import GradientCalls

    if __name__ == "__main__":
        benchmark.benchmark(
            algorithms=[DGD(iterations=1000, step_size=0.001), ADMM(iterations=1000, rho=10, alpha=0.3)],
            benchmark_problem=benchmark_problem.create_regression_problem(LinearRegressionCost),
            table_metrics=[GradientCalls([min, max])],
            plot_metrics=[RegretPerIteration()],
            table_fmt="latex",
            n_trials=10,
            confidence_level=0.9,
            log_level=DEBUG,
            max_processes=1,
        )


Benchmark problems
------------------

Configure out-of-the-box regression problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Configure communication constraints and other settings for out-of-the-box regression problems.

.. code-block:: python

    from decent_bench import benchmark, benchmark_problem
    from decent_bench.costs import LinearRegressionCost
    from decent_bench.distributed_algorithms import ADMM, DGD, ED

    problem = benchmark_problem.create_regression_problem(
        LinearRegressionCost,
        n_agents=100,
        n_neighbors_per_agent=3,
        asynchrony=True,
        compression=True,
        noise=True,
        drops=True,
    )

    if __name__ == "__main__":
        benchmark.benchmark(
            algorithms=[
                DGD(iterations=1000, step_size=0.001),
                ED(iterations=1000, step_size=0.001),
                ADMM(iterations=1000, rho=10, alpha=0.3),
            ],
            benchmark_problem=problem,
        )


Modify existing problems
~~~~~~~~~~~~~~~~~~~~~~~~
Change the settings of an already created benchmark problem, for example, the network topology.

.. code-block:: python

    import networkx as nx

    from decent_bench import benchmark, benchmark_problem
    from decent_bench.costs import LinearRegressionCost
    from decent_bench.distributed_algorithms import ADMM, DGD, ED

    n_agents = 100
    n_neighbors_per_agent = 3

    problem = benchmark_problem.create_regression_problem(
        LinearRegressionCost,
        n_agents=n_agents,
        n_neighbors_per_agent=n_neighbors_per_agent,
        asynchrony=True,
        compression=True,
        noise=True,
        drops=True,
    )

    problem.network_structure = nx.random_regular_graph(n_agents, n_neighbors_per_agent)

    if __name__ == "__main__":
        benchmark.benchmark(
            algorithms=[
                DGD(iterations=1000, step_size=0.001),
                ED(iterations=1000, step_size=0.001),
                ADMM(iterations=1000, rho=10, alpha=0.3),
            ],
            benchmark_problem=problem,
        )


Create problems using existing resources
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Create a custom benchmark problem using existing resources.

.. code-block:: python

    import networkx as nx

    from decent_bench import benchmark
    from decent_bench import centralized_algorithms as ca
    from decent_bench.benchmark_problem import BenchmarkProblem
    from decent_bench.costs import LogisticRegressionCost
    from decent_bench.datasets import SyntheticClassificationData
    from decent_bench.distributed_algorithms import ADMM, DGD, ED
    from decent_bench.schemes import GaussianNoise, Quantization, UniformActivationRate, UniformDropRate
    from decent_bench.utils.types import SupportedFrameworks

    n_agents = 100

    dataset = SyntheticClassificationData(
        n_classes=2, 
        n_partitions=n_agents, 
        n_samples_per_partition=10, 
        n_features=3, 
        framework=SupportedFrameworks.NUMPY,
    )

    costs = [LogisticRegressionCost(*p) for p in dataset.training_partitions()]

    sum_cost = sum(costs[1:], start=costs[0])
    x_optimal = ca.accelerated_gradient_descent(
        sum_cost, x0=None, max_iter=50000, stop_tol=1e-100, max_tol=1e-16
    )

    problem = BenchmarkProblem(
        network_structure=nx.random_regular_graph(3, n_agents, seed=0),
        costs=costs,
        x_optimal=x_optimal,
        agent_activations=[UniformActivationRate(0.5)] * n_agents,
        message_compression=Quantization(n_significant_digits=4),
        message_noise=GaussianNoise(mean=0, sd=0.001),
        message_drop=UniformDropRate(drop_rate=0.5),
    )

    if __name__ == "__main__":
        benchmark.benchmark(
            algorithms=[
                DGD(iterations=1000, step_size=0.001),
                ED(iterations=1000, step_size=0.001),
                ADMM(iterations=1000, rho=10, alpha=0.3),
            ],
            benchmark_problem=problem,
        )


Create problems from scratch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Create a custom benchmark problem with your own dataset, cost function, and communication schemes by implementing the
corresponding abstracts.

.. code-block:: python

    import networkx as nx

    from decent_bench import benchmark
    from decent_bench import centralized_algorithms as ca
    from decent_bench.benchmark_problem import BenchmarkProblem
    from decent_bench.costs import Cost
    from decent_bench.datasets import Dataset
    from decent_bench.distributed_algorithms import DGD, SimpleGT
    from decent_bench.schemes import AgentActivationScheme, CompressionScheme, DropScheme, NoiseScheme

    class MyDataset(Dataset): ...

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
        benchmark.benchmark(
            algorithms=[DGD(iterations=1000, step_size=0.001), SimpleGT(iterations=1000, step_size=0.001)],
            benchmark_problem=problem,
        )

Interoperability requirement
----------------------------
Decent-Bench is designed to interoperate with multiple array/tensor frameworks (NumPy, PyTorch, JAX, etc.). To keep
algorithms framework-agnostic, always use the interoperability layer :class:`~decent_bench.utils.interoperability`, aliased as
`iop`, and the :class:`~decent_bench.utils.array.Array` wrapper when creating, manipulating, and exchanging values:

- Use :class:`decent_bench.utils.interoperability.zeros` instead of framework-specific constructors (e.g., `np.zeros`, `torch.zeros`). 
    Other examples are :meth:`~decent_bench.utils.interoperability.ones_like`, :meth:`~decent_bench.utils.interoperability.rand_like`, :meth:`~decent_bench.utils.interoperability.randn_like`, etc.
    See :mod:`~decent_bench.utils.interoperability` for a full list of available methods and :mod:`~decent_bench.distributed_algorithms` for examples of usage.
- Avoid calling any framework-specific functions directly within your algorithm. 
    Let the :class:`~decent_bench.costs.Cost` implementations handle framework-specific details for 
    :func:`~decent_bench.costs.Cost.function`, :func:`~decent_bench.costs.Cost.gradient`, :func:`~decent_bench.costs.Cost.hessian`, and :func:`~decent_bench.costs.Cost.proximal`.
- When you need to create a new array/tensor, use the interoperability layer to ensure compatibility with the agent's cost function framework and device.
    If a method to create your specific array/tensor is not available, see the implementation of :attr:`~decent_bench.networks.P2PNetwork.weights` as en example.


Algorithms
----------
Create a new algorithm to benchmark against existing ones. 

**Note**: In order for metrics to work, use :attr:`Agent.x <decent_bench.agents.Agent.x>` to update the local primal
variable. Similarly, in order for the benchmark problem's communication schemes to be applied, use the
:attr:`~decent_bench.networks.P2PNetwork`/ :attr:`~decent_bench.networks.FedNetwork` object to retrieve agents and to send and receive messages.
Be sure to use :meth:`~decent_bench.networks.Network.active_agents` during algorithm runtime, so that asynchrony is properly handled.

.. code-block:: python

    import decent_bench.utils.interoperability as iop

    from decent_bench import benchmark, benchmark_problem
    from decent_bench.costs import LinearRegressionCost
    from decent_bench.distributed_algorithms import ADMM, DGD, Algorithm
    from decent_bench.networks import P2PNetwork

    class MyNewAlgorithm(Algorithm):
        iterations: int
        step_size: float
        name: str = "MNA"

        def run(self, network: P2PNetwork) -> None:
            # Initialize agents with Array values using the interoperability layer
            for agent in network.agents():
                x0 = iop.zeros(shape=agent.cost.shape, framework=agent.cost.framework, device=agent.cost.device)
                y0 = iop.zeros(shape=agent.cost.shape, framework=agent.cost.framework, device=agent.cost.device)
                neighbors = network.neighbors(agent)
                agent.initialize(x=x0, received_msgs=dict.fromkeys(neighbors, x0), aux_vars={"y": y0})

            # Run iterations
            W = network.weights
            for k in range(self.iterations):
                for i in network.active_agents(k):
                    i.aux_vars["y_new"] = i.x - self.step_size * i.cost.gradient(i.x)
                    s = iop.stack([W[i, j] * x_j for j, x_j in i.messages.items()])
                    neighborhood_avg = iop.sum(s, axis=0)
                    neighborhood_avg += W[i, i] * i.x
                    i.x = i.aux_vars["y_new"] - i.aux_vars["y"] + neighborhood_avg
                    i.aux_vars["y"] = i.aux_vars["y_new"]
                for i in network.active_agents(k):
                    network.broadcast(i, i.x)
                for i in network.active_agents(k):
                    network.receive_all(i)

    if __name__ == "__main__":
        benchmark.benchmark(
            algorithms=[
                MyNewAlgorithm(iterations=1000, step_size=0.001),
                DGD(iterations=1000, step_size=0.001),
                ADMM(iterations=1000, rho=10, alpha=0.3),
            ],
            benchmark_problem=benchmark_problem.create_regression_problem(LinearRegressionCost),
        )


Metrics
-------
Create your own metrics to tabulate and/or plot.

.. code-block:: python

    import numpy.linalg as la
    import decent_bench.utils.interoperability as iop

    from decent_bench import benchmark, benchmark_problem
    from decent_bench.agents import AgentMetricsView
    from decent_bench.benchmark_problem import BenchmarkProblem
    from decent_bench.costs import LinearRegressionCost
    from decent_bench.distributed_algorithms import ADMM, DGD
    from decent_bench.metrics.plot_metrics import DEFAULT_PLOT_METRICS, PlotMetric, X, Y
    from decent_bench.metrics.table_metrics import DEFAULT_TABLE_METRICS, TableMetric

    def x_error_at_iter(agent: AgentMetricsView, problem: BenchmarkProblem, i: int = -1) -> float:
        # Convert Array values to numpy for custom metric computation
        return float(la.norm(iop.to_numpy(problem.optimal_x) - iop.to_numpy(agent.x_per_iteration[i])))

    class XError(TableMetric):
        description: str = "x error"

        def get_data_from_trial(
            self, agents: list[AgentMetricsView], problem: BenchmarkProblem
        ) -> list[float]:
            return [x_error_at_iter(a, problem) for a in agents]

    class MaxXErrorPerIteration(PlotMetric):
        x_label: str = "iteration"
        y_label: str = "max x error"

        def get_data_from_trial(
            self, agents: list[AgentMetricsView], problem: BenchmarkProblem
        ) -> list[tuple[X, Y]]:
            iter_reached_by_all = min(len(a.x_per_iteration) for a in agents)
            res: list[tuple[X, Y]] = []
            for i in range(iter_reached_by_all):
                y = max([x_error_at_iter(a, problem, i) for a in agents])
                res.append((i, y))
            return res

    if __name__ == "__main__":
        benchmark.benchmark(
            algorithms=[
                DGD(iterations=1000, step_size=0.001),
                ADMM(iterations=1000, rho=10, alpha=0.3),
            ],
            benchmark_problem=benchmark_problem.create_regression_problem(LinearRegressionCost),
            table_metrics=DEFAULT_TABLE_METRICS + [XError([min, max])],
            plot_metrics=DEFAULT_PLOT_METRICS + [MaxXErrorPerIteration()],
        )


Cost Functions
--------------
Create new cost functions by subclassing :class:`~decent_bench.costs.Cost` and using interoperability decorators to keep
your implementation framework-agnostic. The decorators automatically wrap inputs/outputs as `Array` and ensure
compatibility with the selected framework and device of your custom cost.

.. code-block:: python

    from numpy import float64
    from numpy.typing import NDArray

    import decent_bench.utils.interoperability as iop
    from decent_bench.costs import Cost, SumCost
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
        def proximal(self, y: NDArray[float64], rho: float) -> NDArray[float64]:
            # Optional: provide a closed-form proximal if available
            # Otherwise you can rely on `centralized_algorithms.proximal_solver`.
            return y  # identity as a placeholder

        def __add__(self, other: Cost) -> Cost:
            # Support addition of costs
            if self.shape != other.shape:
                raise ValueError(f"Mismatching domain shapes: {self.shape} vs {other.shape}")

            return SumCost([self, other])
