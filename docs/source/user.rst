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
    from decent_bench.cost_functions import LinearRegressionCost
    from decent_bench.distributed_algorithms import ADMM, DGD, GT2

    if __name__ == "__main__":
        benchmark.benchmark(
            algorithms=[
                DGD(iterations=1000, step_size=0.001),
                GT2(iterations=1000, step_size=0.001),
                ADMM(iterations=1000, rho=10, alpha=0.3),
            ],
            benchmark_problem=benchmark_problem.create_regression_problem(LinearRegressionCost),
        )


Execution settings
------------------
Configure settings for metrics, trials, statistical confidence level, logging, and multiprocessing.

.. code-block:: python

    from logging import DEBUG

    from decent_bench import benchmark, benchmark_problem
    from decent_bench.cost_functions import LinearRegressionCost
    from decent_bench.distributed_algorithms import ADMM, DGD
    from decent_bench.metrics.plot_metrics import GlobalCostErrorPerIteration
    from decent_bench.metrics.table_metrics import NrGradientCalls

    if __name__ == "__main__":
        benchmark.benchmark(
            algorithms=[DGD(iterations=1000, step_size=0.001), ADMM(iterations=1000, rho=10, alpha=0.3)],
            benchmark_problem=benchmark_problem.create_regression_problem(LinearRegressionCost),
            table_metrics=[NrGradientCalls([min, max])],
            plot_metrics=[GlobalCostErrorPerIteration()],
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
    from decent_bench.cost_functions import LinearRegressionCost
    from decent_bench.distributed_algorithms import ADMM, DGD, GT2

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
                GT2(iterations=1000, step_size=0.001),
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
    from decent_bench.cost_functions import LinearRegressionCost
    from decent_bench.distributed_algorithms import ADMM, DGD, GT2

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

    problem.topology_structure = nx.random_regular_graph(n_agents, n_neighbors_per_agent)

    if __name__ == "__main__":
        benchmark.benchmark(
            algorithms=[
                DGD(iterations=1000, step_size=0.001),
                GT2(iterations=1000, step_size=0.001),
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
    from decent_bench.cost_functions import LogisticRegressionCost
    from decent_bench.datasets import SyntheticClassificationData
    from decent_bench.distributed_algorithms import ADMM, DGD, GT2
    from decent_bench.schemes import GaussianNoise, Quantization, UniformActivationRate, UniformDropRate

    n_agents = 100
    
    dataset = SyntheticClassificationData(
        n_classes=2, n_partitions=n_agents, n_samples_per_partition=10, n_features=3
    )

    costs = [LogisticRegressionCost(*p) for p in dataset.get_training_partitions()]

    sum_cost = sum(costs[1:], start=costs[0])
    optimal_x = ca.accelerated_gradient_descent(
        sum_cost, x0=None, max_iter=50000, stop_tol=1e-100, max_tol=1e-16
    )

    problem = BenchmarkProblem(
        topology_structure=nx.random_regular_graph(3, n_agents, seed=0),
        cost_functions=costs,
        optimal_x=optimal_x,
        agent_activation_schemes=[UniformActivationRate(0.5)] * n_agents,
        compression_scheme=Quantization(n_significant_digits=4),
        noise_scheme=GaussianNoise(mean=0, sd=0.001),
        drop_scheme=UniformDropRate(drop_rate=0.5),
    )

    if __name__ == "__main__":
        benchmark.benchmark(
            algorithms=[
                DGD(iterations=1000, step_size=0.001),
                GT2(iterations=1000, step_size=0.001),
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
    from decent_bench.cost_functions import CostFunction
    from decent_bench.datasets import Dataset
    from decent_bench.distributed_algorithms import DGD, GT1
    from decent_bench.schemes import AgentActivationScheme, CompressionScheme, DropScheme, NoiseScheme

    class MyDataset(Dataset): ...

    class MyCostFunction(CostFunction): ...

    class MyAgentActivationScheme(AgentActivationScheme): ...

    class MyCompressionScheme(CompressionScheme): ...

    class MyNoiseScheme(NoiseScheme): ...

    class MyDropScheme(DropScheme): ...

    n_agents = 100

    costs = [MyCostFunction(*p) for p in MyDataset().get_training_partitions()]

    sum_cost = sum(costs[1:], start=costs[0])
    optimal_x = ca.accelerated_gradient_descent(
        sum_cost, x0=None, max_iter=50000, stop_tol=1e-100, max_tol=1e-16
    )

    problem = BenchmarkProblem(
        topology_structure=nx.random_regular_graph(3, n_agents, seed=0),
        cost_functions=costs,
        optimal_x=optimal_x,
        agent_activation_schemes=[MyAgentActivationScheme()] * n_agents,
        compression_scheme=MyCompressionScheme(),
        noise_scheme=MyNoiseScheme(),
        drop_scheme=MyDropScheme(),
    )

    if __name__ == "__main__":
        benchmark.benchmark(
            algorithms=[DGD(iterations=1000, step_size=0.001), GT1(iterations=1000, step_size=0.001)],
            benchmark_problem=problem,
        )


Algorithms
----------
Create a new algorithm to benchmark against existing ones. 

**Note**: In order for metrics to work, use :attr:`Agent.x <decent_bench.agent.Agent.x>` to update the local primal
variable. Similarly, in order for the benchmark problem's communication schemes to be applied, use the
:attr:`~decent_bench.network.Network` object to retrieve agents and to send and receive messages.

.. code-block:: python

    import numpy as np

    from decent_bench import benchmark, benchmark_problem
    from decent_bench.cost_functions import LinearRegressionCost
    from decent_bench.distributed_algorithms import ADMM, DGD, DstAlgorithm
    from decent_bench.network import Network

    class MyNewAlgorithm(DstAlgorithm):
        name: str = "MNA"

        def __init__(self, iterations: int, step_size: float):
            self.iterations = iterations
            self.step_size = step_size

        def run(self, network: Network) -> None:
            # Initialize agents
            for agent in network.get_all_agents():
                x0 = np.zeros(agent.cost_function.domain_shape)
                y0 = np.zeros(agent.cost_function.domain_shape)
                neighbors = network.get_neighbors(agent)
                agent.initialize(x=x0, received_msgs=dict.fromkeys(neighbors, x0), aux_vars={"y": y0})

            # Run iterations
            W = network.metropolis_weights
            for k in range(self.iterations):
                for i in network.get_active_agents(k):
                    i.aux_vars["y_new"] = i.x - self.step_size * i.cost_function.gradient(i.x)
                    neighborhood_avg = np.sum(
                        [W[i, j] * x_j for j, x_j in i.received_messages.items()], axis=0
                    )
                    neighborhood_avg += W[i, i] * i.x
                    i.x = i.aux_vars["y_new"] - i.aux_vars["y"] + neighborhood_avg
                    i.aux_vars["y"] = i.aux_vars["y_new"]
                for i in network.get_active_agents(k):
                    network.broadcast(i, i.x)
                for i in network.get_active_agents(k):
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

    from decent_bench import benchmark, benchmark_problem
    from decent_bench.agent import AgentMetricsView
    from decent_bench.benchmark_problem import BenchmarkProblem
    from decent_bench.cost_functions import LinearRegressionCost
    from decent_bench.distributed_algorithms import ADMM, DGD
    from decent_bench.metrics.plot_metrics import DEFAULT_PLOT_METRICS, PlotMetric, X, Y
    from decent_bench.metrics.table_metrics import DEFAULT_TABLE_METRICS, TableMetric

    def x_error_at_iter(agent: AgentMetricsView, problem: BenchmarkProblem, i: int = -1) -> float:
        return float(la.norm(problem.optimal_x - agent.x_per_iteration[i]))

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


Output
------
Benchmark executions will have outputs like these:

.. list-table::

   * - .. image:: _static/table.png
          :align: center
     - .. image:: _static/plot.png
          :align: center
