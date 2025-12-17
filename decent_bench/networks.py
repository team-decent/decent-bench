from functools import cached_property
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

import decent_bench.utils.interoperability as iop
from decent_bench.agents import Agent
from decent_bench.benchmark_problem import BenchmarkProblem
from decent_bench.schemes import CompressionScheme, DropScheme, NoiseScheme
from decent_bench.utils.array import Array

if TYPE_CHECKING:
    AgentGraph = nx.Graph[Agent]
else:
    AgentGraph = nx.Graph


class P2PNetwork:
    """
    Peer-to-Peer Network of agents that communicate by sending and receiving messages.

    Args:
        graph: topology defining how the agents are connected
        message_noise: message noise setting
        message_compression: message compression setting
        message_drop: message drops setting

    """

    def __init__(
        self,
        graph: AgentGraph,
        message_noise: NoiseScheme,
        message_compression: CompressionScheme,
        message_drop: DropScheme,
    ):
        self._graph = graph
        self._message_noise = message_noise
        self._message_compression = message_compression
        self._message_drop = message_drop
        self.W: Array | None = None

    def set_weights(self, weights: Array) -> None:
        """
        Set custom consensus weights matrix.

        A simple way to create custom weights is to start using numpy and then
        use :func:`~decent_bench.utils.interoperability.to_array` to convert to an
        :class:`~decent_bench.utils.array.Array` object with the desired framework and device.
        For an example see :func:`~decent_bench.utils.interoperability.zeros`.

        Note:
            If not set, the weights matrix is initialized using the Metropolis-Hastings method.
            Weights will be overwritten if framework or device differ from
            ``Agent.cost.framework`` or ``Agent.cost.device``.

        """
        self.W = weights

    @property
    def weights(self) -> Array:
        """
        Symmetric, doubly stochastic matrix for consensus weights. Initialized using the Metropolis-Hastings method.

        Use ``weights[i, j]`` or ``weights[i.id, j.id]`` to get the weight between agent i and j.
        """
        agents = self.agents()

        if self.W is not None:
            return self.W

        n = len(agents)
        W = np.zeros((n, n))  # noqa: N806
        for i in agents:
            neighbors = self.neighbors(i)
            d_i = len(neighbors)
            for j in neighbors:
                d_j = len(self.neighbors(j))
                W[i, j] = 1 / (1 + max(d_i, d_j))
        for i in agents:
            W[i, i] = 1 - sum(W[i])

        self.W = iop.to_array(W, agents[0].cost.framework, agents[0].cost.device)
        return self.W

    @cached_property
    def adjacency(self) -> Array:
        """
        Adjacency matrix of the network.

        Use ``adjacency[i, j]`` or ``adjacency[i.id, j.id]`` to get the adjacency between agent i and j.
        """
        agents = self.agents()
        n = len(agents)
        A = np.zeros((n, n))  # noqa: N806
        for i in agents:
            for j in self.neighbors(i):
                A[i, j] = 1

        return iop.to_array(A, agents[0].cost.framework, agents[0].cost.device)

    def agents(self) -> list[Agent]:
        """Get all agents in the network."""
        return list(self._graph)

    def neighbors(self, agent: Agent) -> list[Agent]:
        """Get all neighbors of an agent."""
        return list(self._graph[agent])

    def active_agents(self, iteration: int) -> list[Agent]:
        """
        Get all active agents.

        Whether an :class:`~decent_bench.agents.Agent` is active or not at a given time is defined by its
        :class:`~decent_bench.schemes.AgentActivationScheme`.
        """
        return [a for a in self.agents() if a._activation.is_active(iteration)]  # noqa: SLF001

    def send(self, sender: Agent, receiver: Agent, msg: Array) -> None:
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
        if self._message_drop.should_drop():
            sender._n_sent_messages_dropped += 1  # noqa: SLF001
            return
        msg = self._message_compression.compress(msg)
        msg = self._message_noise.make_noise(msg)
        self._graph.edges[sender, receiver][str(receiver.id)] = msg

    def broadcast(self, sender: Agent, msg: Array) -> None:
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
        :attr:`Agent.messages <decent_bench.agents.Agent.messages>`.
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
        :attr:`Agent.messages <decent_bench.agents.Agent.messages>`.
        """
        for neighbor in self._graph.neighbors(receiver):
            self.receive(receiver, neighbor)


def create_distributed_network(problem: BenchmarkProblem) -> P2PNetwork:
    """
    Create a distributed network - a network with peer-to-peer communication only, no coordinator.

    Raises:
        ValueError: if there are less agent activation schemes or cost functions than agents

    """
    n_agents = len(problem.network_structure)
    if len(problem.agent_activations) < n_agents:
        raise ValueError("Insufficient number of agent activation schemes, please provide one per agent")
    if len(problem.costs) < n_agents:
        raise ValueError("Insufficient number of cost functions, please provide one per agent")
    if problem.network_structure.is_directed():
        raise NotImplementedError("Support for directed graphs has not been implemented yet")
    if problem.network_structure.is_multigraph():
        raise NotImplementedError("Support for multi-graphs has not been implemented yet")
    if not nx.is_connected(problem.network_structure):
        raise NotImplementedError("Support for disconnected graphs has not been implemented yet")
    agents = [
        Agent(i, problem.costs[i], problem.agent_activations[i], problem.agent_history_period) for i in range(n_agents)
    ]
    agent_node_map = {node: agents[i] for i, node in enumerate(problem.network_structure.nodes())}
    graph = nx.relabel_nodes(problem.network_structure, agent_node_map)
    return P2PNetwork(
        graph=graph,
        message_noise=problem.message_noise,
        message_compression=problem.message_compression,
        message_drop=problem.message_drop,
    )
