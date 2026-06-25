import matplotlib.pyplot as plt
import networkx as nx

from decent_bench.agents import Agent
from decent_bench.benchmark import create_regression_problem
from decent_bench.networks import P2PNetwork
from decent_bench.utils.network_utils import plot_network

if __name__ == "__main__":

    ## Problem definition ------------------------------------------------
    n_agents = 10

    costs, _, _ = create_regression_problem(n_agents=n_agents)

    ## Create network ----------------------------------------------------
    graph = nx.complete_graph(n_agents)
    network = P2PNetwork(graph=graph, agents=[Agent(cost) for cost in costs])

    ## Draw network ------------------------------------------------------
    fig = plt.figure()
    plot_network(network, ax=fig.gca())
    plt.show()
