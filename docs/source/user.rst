User Guide
==========
This user guide shows you different examples of how to use decent-bench, from simplest to most advanced.


Installation
------------
Requires `Python 3.13+ <https://www.python.org/downloads/>`_

.. code-block:: bash

    pip install decent-bench


Benchmark on sunny case
-----------------------
Benchmark algorithms on a regression problem without any communication constraints.

.. code-block:: python

    from decent_bench import benchmark, benchmark_problem
    from decent_bench.cost_functions import LinearRegressionCost
    from decent_bench.distributed_algorithms import ADMM, DGD, GT2

    if __name__ == '__main__':
        benchmark.benchmark(
            algorithms=[
                DGD(iterations=1000, step_size=0.001),
                GT2(iterations=1000, step_size=0.001),
                ADMM(iterations=1000, rho=10, alpha=0.3),
            ],
            benchmark_problem=benchmark_problem.create_regression_problem(LinearRegressionCost),
        )


Configure problem settings
--------------------------
Configure communication constraints and other benchmark problem settings.

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

    if __name__ == '__main__':
        benchmark.benchmark(
            algorithms=[
                DGD(iterations=1000, step_size=0.001),
                GT2(iterations=1000, step_size=0.001),
                ADMM(iterations=1000, rho=10, alpha=0.3),
            ],
            benchmark_problem=problem,
        )


Configure runtime settings
--------------------------
Configure settings for metrics, table format, trials, statistical confidence level, logging, and multiprocessing.

.. code-block:: python

    from logging import DEBUG

    from decent_bench import benchmark, benchmark_problem
    from decent_bench.cost_functions import LinearRegressionCost
    from decent_bench.distributed_algorithms import ADMM, DGD
    from decent_bench.metrics.plot_metrics import GlobalCostErrorPerIteration
    from decent_bench.metrics.table_metrics import NrGradientCalls

    if __name__ == '__main__':
        benchmark.benchmark(
            algorithms=[DGD(iterations=1000, step_size=0.001), ADMM(iterations=1000, rho=10, alpha=0.3)],
            benchmark_problem=benchmark_problem.create_regression_problem(LinearRegressionCost),
            table_metrics=[NrGradientCalls([min, max])],
            plot_metrics=[GlobalCostErrorPerIteration()],
            table_fmt="latex",
            n_trials=10,
            confidence_level=0.9,
            log_level=DEBUG,
            max_processes=1
        )


Create custom benchmark problem
-------------------------------
Create a custom benchmark problem using existing resources.

.. code-block:: python

    import networkx as nx
    import numpy as np

    from decent_bench import benchmark
    from decent_bench import centralized_algorithms as ca
    from decent_bench.benchmark import BenchmarkProblem
    from decent_bench.cost_functions import LogisticRegressionCost
    from decent_bench.datasets import SyntheticClassificationData
    from decent_bench.distributed_algorithms import ADMM, DGD, GT2
    from decent_bench.metrics.metric_utils import single
    from decent_bench.metrics.plot_metrics import (
        GlobalCostErrorPerIteration, GlobalGradientOptimalityPerIteration
    )
    from decent_bench.metrics.table_metrics import GlobalCostError, NrGradientCalls
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

    if __name__ == '__main__':
        benchmark.benchmark(
            algorithms=[
                DGD(iterations=1000, step_size=0.001),
                GT2(iterations=1000, step_size=0.001),
                ADMM(iterations=1000, rho=10, alpha=0.3),
            ],
            benchmark_problem=problem,
        )


Create custom everything
------------------------
Create your own algorithm, metrics, and benchmark problem (with your custom dataset, cost function, and communication
schemes) by implementing the corresponding abstracts.

.. code-block:: python

    import networkx as nx

    from decent_bench import benchmark
    from decent_bench import centralized_algorithms as ca
    from decent_bench.benchmark_problem import BenchmarkProblem
    from decent_bench.cost_functions import CostFunction
    from decent_bench.datasets import Dataset
    from decent_bench.distributed_algorithms import DstAlgorithm
    from decent_bench.metrics.plot_metrics import PlotMetric
    from decent_bench.metrics.table_metrics import TableMetric
    from decent_bench.schemes import AgentActivationScheme, CompressionScheme, DropScheme, NoiseScheme

    class MyAlgorithm(DstAlgorithm): ...

    class MyTableMetric(TableMetric): ...

    class MyPlotMetric(PlotMetric): ...

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

    if __name__ == '__main__':
        benchmark.benchmark(
            algorithms=[MyAlgorithm()],
            benchmark_problem=problem,
            table_metrics=[MyTableMetric([min, max, sum])],
            plot_metrics=[MyPlotMetric()]
        )


Output
------
Benchmark executions will have outputs like these:

.. list-table::

   * - .. image:: _static/table.png
          :align: center
     - .. image:: _static/plot.png
          :align: center
