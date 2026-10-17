from decent_array.types import Devices, Frameworks

from decent_bench import benchmark
from decent_bench.agents import Agent
from decent_bench.algorithms.federated import FedAvg, Scaffold
from decent_bench.benchmark import configure, create_regression_problem
from decent_bench.networks import FedNetwork

if __name__ == "__main__":

    configure(
        Frameworks.NUMPY,
        Devices.CPU,
        storage_dir="benchmark_results/long_run",
        n_checkpoints=5,  # save 5 evenly spaced checkpoints, including the final iteration
        compression_level=2,
    )

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
            FedAvg(step_size=step, num_local_steps=num_local_steps),
            Scaffold(step_size=step, num_local_steps=num_local_steps),
        ],
        benchmark_problem=problem,
        iterations=num_iter,
        n_trials=10,
        )
