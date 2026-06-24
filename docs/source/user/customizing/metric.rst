Metrics
-------

Customize metrics at two levels:

- post-run table/plot metrics passed to :func:`~decent_bench.benchmark.compute_metrics`
- runtime metrics passed to :func:`~decent_bench.benchmark.benchmark`

You can create your own metrics by subclassing :class:`~decent_bench.metrics.Metric` or
:class:`~decent_bench.metrics.RuntimeMetric`.




Metrics
-------

Table and plot metrics
~~~~~~~~~~~~~~~~~~~~~~
Create your own metrics to tabulate and/or plot.

.. code-block:: python
    
    from collections.abc import Sequence

    import numpy.linalg as la
    import decent_bench.utils.interoperability as iop

    from decent_bench.metrics import utils
    from decent_bench import benchmark
    from decent_bench.agents import AgentMetricsView
    from decent_bench.benchmark import BenchmarkProblem
    from decent_bench.costs import LinearRegressionCost
    from decent_bench.distributed_algorithms import ADMM, DGD
    from decent_bench.metrics import Metric

    class XError(Metric):

        description: str = "x error"

        def compute(  # noqa: D102
            self,
            agents: Sequence[AgentMetricsView],
            problem: BenchmarkProblem,
            iteration: int,
        ) -> list[float]:
            if problem.x_optimal is None:
                return [float("nan") for _ in agents]

            x_optimal_np = iop.to_numpy(problem.x_optimal)

            if iteration == -1:
                return [float(la.norm(x_optimal_np - iop.to_numpy(a.x_history[a.x_history.max()]))) for a in agents]
            return [
                float(la.norm(x_optimal_np - iop.to_numpy(a.x_history[iteration])))
                for a in agents
            ]

    if __name__ == "__main__":
        x_error = XError(
            statistics=[min, max],
            fmt=".4e",
            x_log=False,
            y_log=True,
        )

        benchmark_result = benchmark.benchmark(
            algorithms=[
                DGD(iterations=1000, step_size=0.01),
                ADMM(iterations=1000, penalty=10, relaxation=0.3),
            ],
            benchmark_problem=benchmark.create_regression_problem(LinearRegressionCost),
        )

        metrics_result = benchmark.compute_metrics(benchmark_result, table_metrics=[x_error], plot_metrics=[x_error])
        benchmark.display_metrics(metrics_result)


Runtime metrics
~~~~~~~~~~~~~~~
Create your own runtime metrics to monitor algorithm progress during execution.

Runtime metrics are computed during algorithm execution to provide live feedback for early stopping or monitoring convergence.
Unlike post-hoc table and plot metrics, runtime metrics don't store historical data and are designed to be lightweight.
They are updated at a specified interval and can optionally save plots to disk after execution.

.. code-block:: python

    from collections.abc import Sequence

    import decent_bench.utils.interoperability as iop
    from decent_bench import benchmark
    from decent_bench.agents import Agent
    from decent_bench.benchmark import BenchmarkProblem
    from decent_bench.costs import LinearRegressionCost
    from decent_bench.distributed_algorithms import DGD
    from decent_bench.metrics import RuntimeMetric

    class RuntimeConsensusError(RuntimeMetric):
        """Monitors how well agents agree on their decision variables."""

        description = "Consensus Error"
        x_log = False
        y_log = True

        def compute(self, problem: BenchmarkProblem, agents: Sequence[Agent], iteration: int) -> float:
            # Compute average x across all agents
            x_avg = iop.mean(iop.stack([agent.x for agent in agents]), dim=0)
            
            # Compute average distance from the mean
            errors = [float(iop.norm(agent.x - x_avg)) for agent in agents]
            return sum(errors) / len(agents)

    class RuntimeRegret(RuntimeMetric):
        """Example how to cache computations"""

        description = "Regret"
        x_log = False
        y_log = False

        def compute(self, problem: BenchmarkProblem, agents: Sequence[Agent], iteration: int) -> float:
            if problem.x_optimal is None:
                return float("nan")

            agent_cost = sum(agent.cost.function(agent.x) for agent in agents) / len(agents)

            if hasattr(self, "_cached_optimal_cost"):
                return agent_cost - self._cached_optimal_cost

            # Since x_optimal is fixed for the problem, we can cache the optimal cost after computing it once
            self._cached_optimal_cost: float = sum(agent.cost.function(problem.x_optimal) for agent in agents) / len(agents)

            return agent_cost - self._cached_optimal_cost

    if __name__ == "__main__":
        benchmark_result = benchmark.benchmark(
            algorithms=[DGD(iterations=10000, step_size=0.001)],
            benchmark_problem=benchmark.create_regression_problem(LinearRegressionCost),
            runtime_metrics=[
                RuntimeConsensusError(
                    update_interval=100,  # Compute and plot every 100 iterations
                    save_path="results",  # Save plots to "results" directory during execution
                )
            ],
        )

**Important considerations for runtime metrics:**

- Keep the :meth:`~decent_bench.metrics.RuntimeMetric.compute` method efficient, as it's called during algorithm execution
- Avoid expensive computations that might significantly slow down the algorithm
- The ``update_interval`` parameter controls the trade-off between monitoring granularity and performance overhead
- If ``save_path`` is provided, plots are saved to disk at each update interval
- Runtime metrics are useful for early stopping, detecting divergence, or monitoring specific convergence properties
