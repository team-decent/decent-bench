from decent_array.types import Devices, Frameworks

from decent_bench.agents import Agent
from decent_bench.algorithms.federated import FedAvg
from decent_bench.benchmark import BenchmarkProblem, configure, create_regression_problem
from decent_bench.metrics import metric_library
from decent_bench.networks import FedNetwork
from decent_bench.utils.tuning import grid_search


if __name__ == "__main__":
    configure(Frameworks.NUMPY, Devices.CPU)

    ## Problem definition ------------------------------------------------
    n_agents = 10
    costs, x_optimal, _ = create_regression_problem(n_agents=n_agents)
    network = FedNetwork(clients=[Agent(cost) for cost in costs])
    problem = BenchmarkProblem(network, x_optimal)

    ## Hyperparameter tuning ---------------------------------------------
    best_hyperparams, results = grid_search(
        benchmark_problem=problem,
        algorithm_class=FedAvg,
        tunable_params={
            "step_size": [0.01, 0.1],
            "num_local_steps": [1, 5],
        },
        iterations=250,
        metric=metric_library.XError(),
        n_trials=1,
    )

    ## Display results ---------------------------------------------------
    print(f"Best hyperparameters: {best_hyperparams}")
    print(results.sort_values("mean").to_string(index=False))
