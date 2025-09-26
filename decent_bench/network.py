from functools import cached_property
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from networkx import Graph
from numpy import float64
from numpy.typing import NDArray

from decent_bench.agent import Agent
from decent_bench.benchmark_problem import BenchmarkProblem
from decent_bench.schemes import CompressionScheme, DropScheme, NoiseScheme

if TYPE_CHECKING:
    AgentGraph = Graph[Agent]
else:
    AgentGraph = Graph


class Network:
    """
    Network of agents that communicate by sending and receiving messages.

    Args:
        graph: topology defining how the agents are connected
        noise_scheme: message noise setting
        compression_scheme: message compression setting
        drop_scheme: message drops setting

    """

    def __init__(
        self,
        graph: AgentGraph,
        noise_scheme: NoiseScheme,
        compression_scheme: CompressionScheme,
        drop_scheme: DropScheme,
    ):
        self._graph = graph
        self._noise_scheme = noise_scheme
        self._compression_scheme = compression_scheme
        self._drop_scheme = drop_scheme

    @cached_property
    def metropolis_weights(self) -> NDArray[float64]:
        """
        Symmetric, doubly stochastic matrix for consensus weights.

        Use ``metropolis_weights[i, j]`` or ``metropolis_weights[i.id, j.id]`` to get the weight between agent i and j.
        """
        agents = self.get_all_agents()
        n = len(agents)
        W = np.zeros((n, n))  # noqa: N806
        for i in agents:
            neighbors = self.get_neighbors(i)
            d_i = len(neighbors)
            for j in neighbors:
                d_j = len(self.get_neighbors(j))
                W[i, j] = 1 / (1 + max(d_i, d_j))
        for i in agents:
            W[i, i] = 1 - sum(W[i])
        return W

    def get_all_agents(self) -> list[Agent]:
        """Get all agents in the network."""
        return list(self._graph)

    def get_neighbors(self, agent: Agent) -> list[Agent]:
        """Get all neighbors of an agent."""
        return list(self._graph[agent])

    def get_active_agents(self, iteration: int) -> list[Agent]:
        """
        Get all active agents.

        Whether an :class:`~decent_bench.agent.Agent` is active or not at a given time is defined by its
        :class:`~decent_bench.schemes.AgentActivationScheme`.
        """
        return [a for a in self.get_all_agents() if a._activation_scheme.is_active(iteration)]  # noqa: SLF001

    def send(self, sender: Agent, receiver: Agent, msg: NDArray[float64]) -> None:
        """
        Send message to a neighbor.

        The message may be compressed, distorted by noise, and/or dropped depending on the network's
        :class:`~decent_bench.schemes.CompressionScheme`,
        :class:`~decent_bench.schemes.NoiseScheme`,
        and :class:`~decent_bench.schemes.DropScheme`.

        The message will stay in-flight until it is received or replaced by a newer message from the same sender to the
        same receiver. After being received or replaced, the message is destroyed.
        """
        sender._n_sent_messages += 1  # noqa: SLF001
        if self._drop_scheme.should_drop():
            sender._n_sent_messages_dropped += 1  # noqa: SLF001
            return
        msg = self._compression_scheme.compress(msg)
        msg = self._noise_scheme.make_noise(msg)
        self._graph.edges[sender, receiver][str(receiver.id)] = msg

    def broadcast(self, sender: Agent, msg: NDArray[float64]) -> None:
        """
        Send message to all neighbors.

        The message may be compressed, distorted by noise, and/or dropped depending on the network's
        :class:`~decent_bench.schemes.CompressionScheme`,
        :class:`~decent_bench.schemes.NoiseScheme`,
        and :class:`~decent_bench.schemes.DropScheme`.

        The message will stay in-flight until it is received or replaced by a newer message from the same sender to the
        same receiver. After being received or replaced, the message is destroyed.
        """
        for neighbor in self._graph.neighbors(sender):
            self.send(sender=sender, receiver=neighbor, msg=msg)

    def receive(self, receiver: Agent, sender: Agent) -> None:
        """
        Receive message from a neighbor.

        Received messages are stored in
        :attr:`Agent.received_messages <decent_bench.agent.Agent.received_messages>`.
        """
        msg = self._graph.edges[sender, receiver].get(str(receiver.id))
        if msg is not None:
            receiver._n_received_messages += 1  # noqa: SLF001
            receiver._received_messages[sender] = msg  # noqa: SLF001
            self._graph.edges[sender, receiver][str(receiver.id)] = None

    def receive_all(self, receiver: Agent) -> None:
        """
        Receive messages from all neighbors.

        Received messages are stored in
        :attr:`Agent.received_messages <decent_bench.agent.Agent.received_messages>`.
        """
        for neighbor in self._graph.neighbors(receiver):
            self.receive(receiver, neighbor)


def create_distributed_network(problem: BenchmarkProblem) -> Network:
    """
    Create a distributed network - a network with peer-to-peer communication only, no coordinator.

    Raises:
        ValueError: if there are less agent activation schemes or cost functions than agents

    """
    n_agents = len(problem.topology_structure)
    if len(problem.agent_activation_schemes) < n_agents:
        raise ValueError("Insufficient number of agent activation schemes, please provide one per agent")
    if len(problem.cost_functions) < n_agents:
        raise ValueError("Insufficient number of cost functions, please provide one per agent")
    if problem.topology_structure.is_directed():
        raise NotImplementedError("Support for directed graphs has not been implemented yet")
    if problem.topology_structure.is_multigraph():
        raise NotImplementedError("Support for multi-graphs has not been implemented yet")
    if not nx.is_connected(problem.topology_structure):
        raise NotImplementedError("Support for disconnected graphs has not been implemented yet")
    agents = [Agent(i, problem.cost_functions[i], problem.agent_activation_schemes[i]) for i in range(n_agents)]
    agent_node_map = {node: agents[i] for i, node in enumerate(problem.topology_structure.nodes())}
    graph = nx.relabel_nodes(problem.topology_structure, agent_node_map)
    return Network(
        graph=graph,
        noise_scheme=problem.noise_scheme,
        compression_scheme=problem.compression_scheme,
        drop_scheme=problem.drop_scheme,
    )
