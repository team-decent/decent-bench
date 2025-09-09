from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from networkx import Graph
from numpy import float64
from numpy.typing import NDArray

from decent_bench.library.core.agent import Agent
from decent_bench.library.core.benchmark_problem.schemes.compression_schemes import CompressionScheme
from decent_bench.library.core.benchmark_problem.schemes.drop_schemes import DropScheme
from decent_bench.library.core.benchmark_problem.schemes.noise_schemes import NoiseScheme

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
        self._drops = 0

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

        Whether an :class:`~decent_bench.library.core.agent.Agent` is active or not at a given time is defined by its
        :class:`~decent_bench.library.core.benchmark_problem.schemes.agent_activation_schemes.AgentActivationScheme`.
        """
        return [a for a in self.get_all_agents() if a._activation_scheme.is_active(iteration)]  # noqa: SLF001

    def send(self, sender: Agent, receiver: Agent, msg: NDArray[float64]) -> None:
        """
        Send message to a neighbor.

        The message may be compressed, distorted by noise, and/or dropped depending on the network's
        :class:`~decent_bench.library.core.benchmark_problem.schemes.compression_schemes.CompressionScheme`,
        :class:`~decent_bench.library.core.benchmark_problem.schemes.noise_schemes.NoiseScheme`,
        and :class:`~decent_bench.library.core.benchmark_problem.schemes.drop_schemes.DropScheme`.

        The message will stay in-flight until it is received or replaced by a newer message from the same sender to the
        same receiver. After being received or replaced, the message is destroyed.
        """
        sender._n_messages_sent += 1  # noqa: SLF001
        if self._drop_scheme.should_drop():
            self._drops += 1
            return
        msg = self._compression_scheme.compress(msg)
        msg = self._noise_scheme.make_noise(msg)
        self._graph.edges[sender, receiver][str(receiver.id)] = msg

    def broadcast(self, sender: Agent, msg: NDArray[float64]) -> None:
        """
        Send message to all neighbors.

        The message may be compressed, distorted by noise, and/or dropped depending on the network's
        :class:`~decent_bench.library.core.benchmark_problem.schemes.compression_schemes.CompressionScheme`,
        :class:`~decent_bench.library.core.benchmark_problem.schemes.noise_schemes.NoiseScheme`,
        and :class:`~decent_bench.library.core.benchmark_problem.schemes.drop_schemes.DropScheme`.

        The message will stay in-flight until it is received or replaced by a newer message from the same sender to the
        same receiver. After being received or replaced, the message is destroyed.
        """
        for neighbor in self._graph.neighbors(sender):
            self.send(sender=sender, receiver=neighbor, msg=msg)

    def receive(self, receiver: Agent, sender: Agent) -> None:
        """
        Receive message from a neighbor.

        Received messages are stored in
        :attr:`Agent.received_messages <decent_bench.library.core.agent.Agent.received_messages>`.
        """
        msg = self._graph.edges[sender, receiver].get(str(receiver.id))
        if msg is not None:
            receiver._n_messages_received += 1  # noqa: SLF001
            receiver._received_messages[sender] = msg  # noqa: SLF001
            self._graph.edges[sender, receiver][str(receiver.id)] = None

    def receive_all(self, receiver: Agent) -> None:
        """
        Receive messages from all neighbors.

        Received messages are stored in
        :attr:`Agent.received_messages <decent_bench.library.core.agent.Agent.received_messages>`.
        """
        for neighbor in self._graph.neighbors(receiver):
            self.receive(receiver, neighbor)
