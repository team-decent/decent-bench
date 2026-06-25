import networkx as nx

from decent_bench.agents import Agent
from decent_bench.benchmark import create_regression_problem
from decent_bench.networks import P2PNetwork

if __name__ == "__main__":

    ## Problem definition ------------------------------------------------
    n_agents = 10

    costs, _, _ = create_regression_problem(n_agents=n_agents)

    ## Complete topology -------------------------------------------------
    graph = nx.complete_graph(n_agents)
    network = P2PNetwork(graph=graph, agents=[Agent(cost) for cost in costs])

    ## Star topology -----------------------------------------------------
    graph = nx.star_graph(n_agents-1)
    network = P2PNetwork(graph=graph, agents=[Agent(cost) for cost in costs])

    ## Ring topology -----------------------------------------------------
    graph = nx.cycle_graph(n_agents)
    network = P2PNetwork(graph=graph, agents=[Agent(cost) for cost in costs])
