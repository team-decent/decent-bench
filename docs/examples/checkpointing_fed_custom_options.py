import platform

from decent_bench import benchmark
from decent_bench.agents import Agent
from decent_bench.algorithms.federated import FedAvg, Scaffold
from decent_bench.benchmark import create_regression_problem
from decent_bench.networks import FedNetwork
from decent_bench.utils.checkpoint_manager import CheckpointManager

if __name__ == "__main__":

    cm = CheckpointManager(
        checkpoint_dir="benchmark_results/long_run",
        checkpoint_step=50,      # checkpoint every 50 iterations
        keep_n_checkpoints=5,    # keep 5 most recent checkpoints
        benchmark_metadata={
            "description": "FedAvg v. Scaffold",
            "system": platform.system(),
            "python_version": platform.python_version(),
            "notes": "step = 0.1, num_local_steps = 10",
        },
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
            FedAvg(iterations=num_iter, step_size=step, num_local_steps=num_local_steps),
            Scaffold(iterations=num_iter, step_size=step, num_local_steps=num_local_steps),
        ],
        benchmark_problem=problem,
        n_trials=10,
        checkpoint_manager=cm,
        )
