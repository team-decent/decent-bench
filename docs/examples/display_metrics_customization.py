from decent_bench import benchmark
from decent_bench.agents import Agent
from decent_bench.algorithms.federated import FedAvg, Scaffold
from decent_bench.benchmark import create_regression_problem
from decent_bench.metrics import ComputationalCost
from decent_bench.networks import FedNetwork

if __name__ == "__main__":

    ## Problem definition ------------------------------------------------
    n_agents = 10

    costs, x_optimal, _ = create_regression_problem(n_agents=n_agents)
    network = FedNetwork(clients=[Agent(cost) for cost in costs])
    problem = benchmark.BenchmarkProblem(network, x_optimal)

    ## Benchmarking ------------------------------------------------------
    num_iter = 250
    step = 0.1
    num_local_steps = 10

    results = benchmark.benchmark(
        algorithms=[
            FedAvg(iterations=num_iter, step_size=step, num_local_steps=num_local_steps),
            Scaffold(iterations=num_iter, step_size=step, num_local_steps=num_local_steps),
        ],
        benchmark_problem=problem,
        n_trials=1,
        )

    ## Computing results -------------------------------------------------
    metrics_results = benchmark.compute_metrics(
        benchmark_result=results,
        table_metrics=None,  # use all available metrics
        plot_metrics=None,  # use all available metrics
        statistics_across_agents=["mean", "std"],  # compute mean and std values of metrics across agents
        )

    ## Displaying results ------------------------------------------------
    # 1) display with default options
    benchmark.display_metrics(metrics_results)

    # 2) display subset of plots, with iteration and computational cost side-by-side
    computational_cost = ComputationalCost(function=1, gradient=2, hessian=10, proximal=20, communication=1)
    benchmark.display_metrics(
        metrics_results,
        table_metrics=None,  # print all
        plot_metrics=["x error", "gradient norm"],  # only two metrics
        algorithms=["Scaffold"],  # only one algorithm
        computational_cost=computational_cost,
        compare_iterations_and_computational_cost=True,  # two plots side-by-side for each metric
        )

    # 3) display only table in LaTeX format
    benchmark.display_metrics(
        metrics_results,
        table_metrics=None,  # print all
        plot_metrics=[],  # no plots
        table_fmt="latex",
        )
