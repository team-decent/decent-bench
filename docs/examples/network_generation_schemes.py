import networkx as nx

from decent_bench.agents import Agent
from decent_bench.benchmark import create_regression_problem
from decent_bench.networks import P2PNetwork
from decent_bench.schemes import GaussianNoise, MarkovChainActivation, Quantization, UniformDropRate

if __name__ == "__main__":

    ## Problem definition ------------------------------------------------
    n_agents = 10

    costs, _, _ = create_regression_problem(n_agents=n_agents)

    ## Agents ------------------------------------------------------------
    agents = [
        Agent(cost, activation=MarkovChainActivation()) for cost in costs
    ]

    ## Network -----------------------------------------------------------
    graph = nx.complete_graph(n_agents)

    network = P2PNetwork(
        graph=graph,
        agents=agents,
        message_noise=GaussianNoise(mean=0.5, std=1),
        message_compression=Quantization(quantization_step=0.1),
        message_drop=UniformDropRate(drop_rate=0.75),
    )
