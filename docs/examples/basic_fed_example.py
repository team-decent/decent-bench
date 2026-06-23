from decent_bench import benchmark
from decent_bench.agents import Agent
from decent_bench.algorithms.federated import FedAvg, Scaffold
from decent_bench.benchmark import create_quadratic_problem
from decent_bench.metrics import metric_library
from decent_bench.networks import FedNetwork

if __name__ == "__main__":

    ## Problem definition ------------------------------------------------
    n_agents = 10

    costs, x_optimal = create_quadratic_problem(size=10, n_agents=n_agents)
    network = FedNetwork(clients=[Agent(cost) for cost in costs])
    problem = benchmark.BenchmarkProblem(network, x_optimal)

    ## Benchmarking ------------------------------------------------------
    num_iter = 250
    step = 0.01
    num_local_steps = 10

    results = benchmark.benchmark(
        algorithms=[
            FedAvg(iterations=num_iter, step_size=step, num_local_steps=num_local_steps),
            Scaffold(iterations=num_iter, step_size=step, num_local_steps=num_local_steps),
        ],
        benchmark_problem=problem,
        n_trials=1,
        )

    ## Computing & displaying results ------------------------------------
    metrics_to_compute = [metric_library.XError(), metric_library.GradientNorm()]

    metrics_results = benchmark.compute_metrics(
        benchmark_result=results,
        table_metrics=metrics_to_compute,
        plot_metrics=metrics_to_compute,
        )

    benchmark.display_metrics(metrics_results)
