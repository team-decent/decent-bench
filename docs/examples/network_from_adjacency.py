import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from decent_bench.agents import Agent
from decent_bench.benchmark import create_regression_problem
from decent_bench.networks import P2PNetwork
from decent_bench.utils.network_utils import plot_network

if __name__ == "__main__":

    ## Problem definition ------------------------------------------------
    n_agents = 10

    costs, _, _ = create_regression_problem(n_agents=n_agents)

    ## Create network ----------------------------------------------------
    adj_mat = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 1, 1, 1, 0, 0, 0],
                        [0, 1, 0, 1, 0, 0, 0, 1, 1, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
                        ])

    graph = nx.from_numpy_array(adj_mat)
    network = P2PNetwork(graph=graph, agents=[Agent(cost) for cost in costs])

    ## Draw network ------------------------------------------------------
    fig = plt.figure()
    plot_network(network, ax=fig.gca())
    plt.show()
