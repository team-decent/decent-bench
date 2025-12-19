from abc import ABC
from collections.abc import Iterable, Mapping
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


class Network(ABC):  # noqa: B024
    """Base network object defining communication logic shared by all network types."""

    def __init__(
        self,
        graph: AgentGraph,
        message_noise: NoiseScheme,
        message_compression: CompressionScheme,
        message_drop: DropScheme,
    ) -> None:
        self._graph = graph
        self._message_noise = message_noise
        self._message_compression = message_compression
        self._message_drop = message_drop

    @property
    def graph(self) -> AgentGraph:
        """Underlying NetworkX graph; mutating it will change the network."""
        return self._graph

    @property
    def G(self) -> AgentGraph:  # noqa: N802
        """Alias for the underlying graph."""
        return self.graph

    def agents(self) -> list[Agent]:
        """Get all agents in the network."""
        return list(self.graph)

    def active_agents(self, iteration: int) -> list[Agent]:
        """
        Get all active agents.

        Whether an :class:`~decent_bench.agents.Agent` is active or not at a given time is defined by its
        :class:`~decent_bench.schemes.AgentActivationScheme`.
        """
        return [a for a in self.agents() if a._activation.is_active(iteration)]  # noqa: SLF001

    def connected_agents(self, agent: Agent) -> list[Agent]:
        """Agents directly connected to ``agent`` in the underlying graph."""
        return list(self.graph.neighbors(agent))

    def _send_one(self, sender: Agent, receiver: Agent, msg: Array) -> None:
        """
        Send message to an agent.

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
        self.graph.edges[sender, receiver][str(receiver.id)] = msg

    def send(
        self,
        sender: Agent,
        receiver: Agent | Iterable[Agent] | None = None,
        msg: Array | None = None,
    ) -> None:
        """
        Send message to one or more agents.

        Args:
            sender: sender agent
            receiver: receiver agent, iterable of receiver agents, or ``None`` to broadcast to connected agents.
            msg: message to send

        Raises:
            ValueError: if ``msg`` is not provided, if agents are not part of the network, or if sender/receiver are not
                connected.

        """
        if msg is None:
            raise ValueError("msg must be provided")

        if sender not in self.graph:
            raise ValueError("Sender must be an agent in the network")

        receivers: Iterable[Agent] | list[Agent]
        if receiver is None:
            receivers = self.connected_agents(sender)
        elif isinstance(receiver, Agent):
            receivers = [receiver]
        else:
            receivers = receiver

        receivers = list(receivers)
        if any(r not in self.connected_agents(sender) for r in receivers):
            raise ValueError("Sender and receiver must be connected in the network")

        for r in receivers:
            self._send_one(sender=sender, receiver=r, msg=msg)

    def _receive_one(self, receiver: Agent, sender: Agent) -> None:
        """
        Receive message from an agent.

        Received messages are stored in
        :attr:`Agent.messages <decent_bench.agents.Agent.messages>`.
        """
        msg = self.graph.edges[sender, receiver].get(str(receiver.id))
        if msg is not None:
            receiver._n_received_messages += 1  # noqa: SLF001
            receiver._received_messages[sender] = msg  # noqa: SLF001
            self.graph.edges[sender, receiver][str(receiver.id)] = None

    def receive(self, receiver: Agent, sender: Agent | Iterable[Agent] | None = None) -> None:
        """
        Receive message(s) at an agent.

        Args:
            receiver: receiver agent
            sender: sender agent, iterable of sender agents, or ``None`` to receive from all connected agents.

        Raises:
            ValueError: if sender/receiver are not part of the network or not connected.

        """
        if receiver not in self.graph:
            raise ValueError("Receiver must be an agent in the network")

        senders: Iterable[Agent] | list[Agent]
        if sender is None:
            senders = self.connected_agents(receiver)
        elif isinstance(sender, Agent):
            senders = [sender]
        else:
            senders = sender

        senders = list(senders)
        if any(s not in self.connected_agents(receiver) for s in senders):
            raise ValueError("Sender and receiver must be connected in the network")

        for s in senders:
            self._receive_one(receiver=receiver, sender=s)


class P2PNetwork(Network):
    """Peer-to-peer network architecture where agents communicate directly with each other."""

    def __init__(
        self,
        graph: AgentGraph,
        message_noise: NoiseScheme,
        message_compression: CompressionScheme,
        message_drop: DropScheme,
    ) -> None:
        super().__init__(
            graph=graph,
            message_noise=message_noise,
            message_compression=message_compression,
            message_drop=message_drop,
        )
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

    def neighbors(self, agent: Agent) -> list[Agent]:
        """Alias for :meth:`~decent_bench.networks.Network.connected_agents`."""
        return super().connected_agents(agent)

    def broadcast(self, sender: Agent, msg: Array) -> None:
        """Send to all neighbors (alias for :meth:`~decent_bench.networks.Network.send` with ``receiver=None``)."""
        self.send(sender=sender, receiver=None, msg=msg)

    def receive_all(self, receiver: Agent) -> None:
        """Receive from all neighbors (alias for Network.receive with sender=None)."""
        self.receive(receiver=receiver, sender=None)


class FedNetwork(Network):
    """Federated learning network with one server node connected to all client nodes (star topology)."""

    def __init__(
        self,
        graph: AgentGraph,
        message_noise: NoiseScheme,
        message_compression: CompressionScheme,
        message_drop: DropScheme,
    ) -> None:
        super().__init__(
            graph=graph,
            message_noise=message_noise,
            message_compression=message_compression,
            message_drop=message_drop,
        )
        self._server = self._identify_server()

    def _identify_server(self) -> Agent:
        degrees = dict(self.graph.degree())
        if not degrees:
            raise ValueError("FedNetwork requires at least one agent")
        server, max_degree = max(degrees.items(), key=lambda item: item[1])  # noqa: FURB118
        n = len(degrees)
        if max_degree != n - 1 or any(deg != 1 for node, deg in degrees.items() if node != server):
            raise ValueError("FedNetwork expects a star topology with one server connected to all clients")
        return server

    @property
    def server(self) -> Agent:
        """Agent acting as the central server."""
        return self._server

    @property
    def coordinator(self) -> Agent:
        """Alias for :attr:`server`."""
        return self.server

    def agents(self) -> list[Agent]:
        """Get all client agents (excludes the server/coordinator)."""
        return [agent for agent in self.graph if agent is not self.server]

    def active_agents(self, iteration: int) -> list[Agent]:
        """Get all active client agents (excludes the server/coordinator)."""
        # Delegates to Network.active_agents(), which iterates over self.agents() (clients only for FedNetwork).
        return super().active_agents(iteration)

    @property
    def clients(self) -> list[Agent]:
        """Alias for :meth:`agents`."""
        return self.agents()

    def active_clients(self, iteration: int) -> list[Agent]:
        """Alias for :meth:`active_agents`."""
        return self.active_agents(iteration)

    def send(
        self,
        sender: Agent,
        receiver: Agent | Iterable[Agent] | None = None,
        msg: Array | None = None,
    ) -> None:
        """
        Send message(s) in a federated learning network.

        Only server <-> client communication is allowed. Client-to-client and server-to-server communication will
        raise an error.

        Raises:
            ValueError: if server-to-server or client-to-client communication is attempted, or if a non-server tries to
                send to multiple receivers. Also see :meth:`Network.send` for generic validation.

        """
        if isinstance(receiver, Agent):
            if sender is self.server and receiver is self.server:
                raise ValueError("Server-to-server communication is not supported")
            if sender is not self.server and receiver is not self.server:
                raise ValueError("Client-to-client communication is not supported")
            super().send(sender=sender, receiver=receiver, msg=msg)
            return

        if receiver is None:
            super().send(sender=sender, receiver=receiver, msg=msg)
            return

        receivers = list(receiver)
        if sender is not self.server:
            raise ValueError("Only the server can send to multiple receivers")
        if any(r is self.server for r in receivers):
            raise ValueError("All receivers must be clients")
        super().send(sender=sender, receiver=receivers, msg=msg)

    def receive(self, receiver: Agent, sender: Agent | Iterable[Agent] | None = None) -> None:
        """
        Receive message(s) in a federated learning network.

        Only server <-> client communication is allowed. Client-to-client and server-to-server communication will
        raise an error.

        Raises:
            ValueError: if sender/receiver roles are invalid. Also see :meth:`Network.receive` for generic validation.

        """
        if isinstance(sender, Agent):
            if receiver is self.server and sender is self.server:
                raise ValueError("Server-to-server communication is not supported")
            if receiver is not self.server and sender is not self.server:
                raise ValueError("Client-to-client communication is not supported")
            super().receive(receiver=receiver, sender=sender)
            return

        if sender is None:
            super().receive(receiver=receiver, sender=sender)
            return

        senders = list(sender)
        if receiver is not self.server:
            raise ValueError("Only the server can receive from multiple senders")
        if any(s is self.server for s in senders):
            raise ValueError("All senders must be clients")
        super().receive(receiver=receiver, sender=senders)

    def send_to_client(self, client: Agent, msg: Array) -> None:
        """
        Send a message from the server to a specific client.

        Raises:
            ValueError: if the receiver is not a client.

        """
        if client not in self.clients:
            raise ValueError("Receiver must be a client")
        self.send(sender=self.server, receiver=client, msg=msg)

    def send_to_all_clients(self, msg: Array) -> None:
        """Send the same message from the server to every client (synchronous FL push)."""
        self.send(sender=self.server, receiver=None, msg=msg)

    def send_from_client(self, client: Agent, msg: Array) -> None:
        """
        Send a message from a client to the server.

        Raises:
            ValueError: if the sender is not a client.

        """
        if client not in self.clients:
            raise ValueError("Sender must be a client")
        self.send(sender=client, receiver=self.server, msg=msg)

    def send_from_all_clients(self, msgs: Mapping[Agent, Array]) -> None:
        """
        Send messages from each client to the server (synchronous FL push).

        Args:
            msgs: mapping from client Agent to the message that client should send. Must include all clients.

        Raises:
            ValueError: if any sender is not a client or if any client is missing.

        """
        clients = set(self.clients)
        senders = set(msgs)
        invalid = senders - clients
        if invalid:
            raise ValueError("All senders must be clients")
        missing = clients - senders
        if missing:
            raise ValueError("Messages must be provided for all clients")
        for client, msg in msgs.items():
            self.send_from_client(client, msg)

    def receive_at_client(self, client: Agent) -> None:
        """
        Receive a message at a client from the server.

        Raises:
            ValueError: if the receiver is not a client.

        """
        if client not in self.clients:
            raise ValueError("Receiver must be a client")
        self.receive(receiver=client, sender=None)

    def receive_at_all_clients(self) -> None:
        """Receive messages at every client from the server (synchronous FL pull)."""
        for client in self.clients:
            self.receive_at_client(client)

    def receive_from_client(self, client: Agent) -> None:
        """
        Receive a message at the server from a specific client.

        Raises:
            ValueError: if the sender is not a client.

        """
        if client not in self.clients:
            raise ValueError("Sender must be a client")
        self.receive(receiver=self.server, sender=client)

    def receive_from_all_clients(self) -> None:
        """Receive messages at the server from every client (synchronous FL pull)."""
        self.receive(receiver=self.server, sender=None)


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
    agents = [Agent(i, problem.costs[i], problem.agent_activations[i]) for i in range(n_agents)]
    agent_node_map = {node: agents[i] for i, node in enumerate(problem.network_structure.nodes())}
    graph = nx.relabel_nodes(problem.network_structure, agent_node_map)
    return P2PNetwork(
        graph=graph,
        message_noise=problem.message_noise,
        message_compression=problem.message_compression,
        message_drop=problem.message_drop,
    )


def create_federated_network(problem: BenchmarkProblem) -> FedNetwork:
    """
    Create a federated learning network with a single server and multiple clients (star topology).

    Raises:
        ValueError: if there are fewer activation schemes or cost functions than agents
        ValueError: if the provided graph is not a star (one server connected to all clients)

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
    degrees = dict(problem.network_structure.degree())
    if n_agents:
        server, max_degree = max(degrees.items(), key=lambda item: item[1])  # noqa: FURB118
        if max_degree != n_agents - 1 or any(deg != 1 for node, deg in degrees.items() if node != server):
            raise ValueError("Federated network requires a star topology (one server connected to all clients)")
    agents = [Agent(i, problem.costs[i], problem.agent_activations[i]) for i in range(n_agents)]
    agent_node_map = {node: agents[i] for i, node in enumerate(problem.network_structure.nodes())}
    graph = nx.relabel_nodes(problem.network_structure, agent_node_map)
    return FedNetwork(
        graph=graph,
        message_noise=problem.message_noise,
        message_compression=problem.message_compression,
        message_drop=problem.message_drop,
    )
