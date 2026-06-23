import networkx as nx

from decent_bench import benchmark
from decent_bench.agents import Agent
from decent_bench.algorithms.p2p import ADMM, DGD, ED
from decent_bench.benchmark import create_quadratic_problem
from decent_bench.metrics import metric_library
from decent_bench.networks import P2PNetwork
from decent_bench.utils.checkpoint_manager import CheckpointManager

if __name__ == "__main__":

    cm = CheckpointManager("results")

    ## Problem definition ------------------------------------------------
    n_agents = 10

    costs, x_optimal = create_quadratic_problem(size=10, n_agents=n_agents)

    graph = nx.complete_graph(n_agents)  # topology of the network
    network = P2PNetwork(graph=graph, agents=[Agent(cost) for cost in costs])

    problem = benchmark.BenchmarkProblem(network, x_optimal)

    ## Benchmarking ------------------------------------------------------
    num_iter = 250

    results = benchmark.benchmark(
        algorithms=[
            DGD(iterations=num_iter, step_size=0.01),
            ED(iterations=num_iter, step_size=0.01),
            ADMM(iterations=num_iter, penalty=1, relaxation=0.8),
        ],
        benchmark_problem=problem,
        n_trials=1,
        checkpoint_manager=cm,
        )

    ## Computing & displaying results ------------------------------------
    metrics_to_compute = [metric_library.XError(), metric_library.GradientNorm()]

    metrics_results = benchmark.compute_metrics(
        benchmark_result=results,
        table_metrics=metrics_to_compute,
        plot_metrics=metrics_to_compute,
        checkpoint_manager=cm,
        )

    benchmark.display_metrics(metrics_results, checkpoint_manager=cm)
