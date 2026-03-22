from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Collection, Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

import networkx as nx
import numpy as np

import decent_bench.utils.interoperability as iop
from decent_bench.agents import Agent
from decent_bench.costs import ZeroCost
from decent_bench.schemes import (
    AlwaysActive,
    CompressionScheme,
    DropScheme,
    NoCompression,
    NoDrops,
    NoiseScheme,
    NoNoise,
)
from decent_bench.utils.array import Array

if TYPE_CHECKING:
    AnyGraph = nx.Graph[Any]
    AgentGraph = nx.Graph[Agent]
else:
    AnyGraph = nx.Graph
    AgentGraph = nx.Graph


class Network(ABC):  # noqa: B024
    """
    Base network object defining communication logic shared by all network types.

    Args:
        graph: underlying NetworkX graph defining the network topology.
            Nodes must be of type :class:`~decent_bench.agents.Agent`.
        drop_unread_messages: whether to drop messages that are not received by the next iteration (i.e. messages that
            are "in-flight" for more than one iteration). If ``False``, messages will stay in-flight until they are
            received or replaced by a newer message from the same sender to the same receiver. After being received or
            replaced, the message is destroyed.
        message_noise: noise scheme(s) to apply to messages sent by agents in the network. Can be a single
            :class:`~decent_bench.schemes.NoiseScheme` instance to apply the same scheme to all agents, a dictionary
            mapping each agent to its scheme, or ``None`` to apply no noise to any agent.
        message_compression: compression scheme(s) to apply to messages sent by agents in the network. Can be a single
            :class:`~decent_bench.schemes.CompressionScheme` instance to apply the same scheme to all agents, a
            dictionary mapping each agent to its scheme, or ``None`` to apply no compression to any agent.
        message_drop: drop scheme(s) to apply to messages sent by agents in the network. Can be a single
            :class:`~decent_bench.schemes.DropScheme` instance to apply the same scheme to all agents, a dictionary
            mapping each agent to its scheme, or ``None`` to apply no message drop to any agent.

    """

    def __init__(
        self,
        graph: AgentGraph,
        drop_unread_messages: bool = True,
        message_noise: NoiseScheme | dict[Agent, NoiseScheme] | None = None,
        message_compression: (CompressionScheme | dict[Agent, CompressionScheme] | None) = None,
        message_drop: DropScheme | dict[Agent, DropScheme] | None = None,
    ) -> None:
        # check that graph is connected and not a multi-graph
        if graph.is_multigraph():
            raise NotImplementedError("Support for multi-graphs is not available")
        if graph.is_directed():
            raise ValueError("Directed graphs are not supported; please provide an undirected graph")
        if not nx.is_connected(graph):
            raise ValueError("The graph needs to be connected")
        agent_ids = [agent.id for agent in graph.nodes()]
        if len(agent_ids) != len(set(agent_ids)):
            raise ValueError("Agent IDs must be unique")

        self._graph = graph
        self._message_noise = self._initialize_message_schemes(message_noise, "noise", NoiseScheme, NoNoise)
        self._message_compression = self._initialize_message_schemes(
            message_compression, "compression", CompressionScheme, NoCompression
        )
        self._message_drop = self._initialize_message_schemes(message_drop, "drop", DropScheme, NoDrops)
        self._active_agents_cache: dict[int, list[Agent]] = {}
        self._active_connected_agents_cache: dict[tuple[Agent, int], list[Agent]] = {}
        self._drop_unread_messages = drop_unread_messages
        self._iteration = 0  # Current iteration, updated by the algorithm

    def _initialize_message_schemes(
        self,
        scheme: object,
        scheme_name: str,
        scheme_class: type,
        default_factory: Callable[[], Any] | None = None,
    ) -> dict[Agent, Any]:
        """
        Create dictionary of message schemes.

        Given the value of `scheme`, the method creates the dictionary as follows:
        - `None`: use `default_factory()` for every agent
        - a single `scheme_class` instance: apply it to every agent
        - uses the given dictionary (provided it contains a value for each agent)

        Args:
            scheme: None, a single scheme instance, or a sequence of scheme instances.
            scheme_name: human-readable scheme category for error messages (e.g. "noise").
            scheme_class: expected class for single-object validation.
            default_factory: factory to call per agent when `scheme is None`.

        Returns:
            dict[Agent, scheme]: mapping each agent in `self.graph` to its scheme instance.

        Raises:
            ValueError: if `scheme` is a sequence and length != number of agents in network.

        """
        if scheme is None:  # no scheme, use default
            if default_factory is None:
                raise ValueError(f"default_factory must be provided for {scheme_name}")
            return {agent: default_factory() for agent in self.graph}
        if isinstance(scheme, scheme_class):  # one scheme, use for all agents
            return dict.fromkeys(self.graph, scheme)
        if isinstance(scheme, dict):  # one scheme per agent
            for agent in self.graph:
                if agent not in scheme:
                    raise ValueError(f"{scheme_name} scheme not provided for agent {agent}")
            return {agent: scheme[agent] for agent in self.graph}
        raise ValueError(
            f"Invalid {scheme_name} scheme: expected None, a {scheme_class.__name__} instance, "
            f" or a dict, got {type(scheme)}",
        )

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
        return self._agents_cache

    @cached_property
    def _agents_cache(self) -> list[Agent]:
        """Cached list of agents; assumes the underlying graph is not mutated after construction."""
        return list(self.graph)

    @property
    def degrees(self) -> dict[Agent, int]:
        """Degree of each agent in the network."""
        return dict(self.graph.degree())

    @property
    def edges(self) -> list[tuple[Agent, Agent]]:
        """Edges of the network as (agent, agent) tuples."""
        return list(self.graph.edges())

    def active_agents(self, iteration: int) -> list[Agent]:
        """
        Get all active agents.

        Whether an :class:`~decent_bench.agents.Agent` is active or not at a given time is defined by its
        :class:`~decent_bench.schemes.AgentActivationScheme`.
        """
        if iteration not in self._active_agents_cache:
            # Cache the active agents for each iterations in case the same iteration is queried multiple times
            # so that we preserve the activated agents for the iteration even if the underlying activation schemes
            # are non-deterministic.
            self._active_agents_cache[iteration] = [a for a in self.agents() if a._activation.is_active(iteration)]  # noqa: SLF001
        return self._active_agents_cache[iteration]

    def connected_agents(self, agent: Agent) -> list[Agent]:
        """Agents directly connected to ``agent`` in the underlying graph."""
        return list(self.graph.neighbors(agent))

    def active_connected_agents(self, agent: Agent, iteration: int) -> list[Agent]:
        """Agents directly connected to ``agent`` and are active at the given iteration."""
        key = (agent, iteration)
        if key not in self._active_connected_agents_cache:
            active_agents = set(self.active_agents(iteration))
            self._active_connected_agents_cache[key] = [a for a in self.connected_agents(agent) if a in active_agents]
        return self._active_connected_agents_cache[key]

    def _send_one(self, sender: Agent, receiver: Agent, msg: Array) -> None:
        """
        Send message to an agent.

        The message may be compressed, distorted by noise, and/or dropped depending on the network's
        :class:`~decent_bench.schemes.CompressionScheme`,
        :class:`~decent_bench.schemes.NoiseScheme`,
        and :class:`~decent_bench.schemes.DropScheme`.

        The message will be emmidiately available to the receiver if it is active in the current iteration.
        """
        sender._n_sent_messages += 1  # noqa: SLF001
        if self._message_drop[sender].should_drop():
            sender._n_sent_messages_dropped += 1  # noqa: SLF001
            return
        msg = self._message_compression[sender].compress(msg)
        msg = self._message_noise[sender].make_noise(msg)
        receiver._n_received_messages += 1  # noqa: SLF001
        receiver._received_messages[sender] = msg  # noqa: SLF001

    def send(
        self,
        sender: Agent,
        receiver: Agent | Sequence[Agent] | None = None,
        msg: Array | None = None,
    ) -> None:
        """
        Send message to one or more agents.

        Args:
            sender: sender agent
            receiver: receiver agent, sequence of receiver agents, or ``None`` to broadcast to connected agents.
            msg: message to send

        Raises:
            ValueError: if ``msg`` is not provided, if agents are not part of the network, or if sender/receiver are not
                connected.

        """
        if msg is None:
            raise ValueError("msg must be provided")

        if sender not in self.graph:
            raise ValueError("Sender must be an agent in the network")

        if receiver is None:
            receiver = self.active_connected_agents(sender, self._iteration)
        elif isinstance(receiver, Agent):
            if receiver not in self.connected_agents(sender):
                raise ValueError("Sender and receiver must be connected in the network")
            self._send_one(sender=sender, receiver=receiver, msg=msg)
            return
        else:
            # Its a sequence of agents, check that all are connected to sender before sending any messages
            neighbors = set(self.connected_agents(sender))
            invalid_receivers = [r for r in receiver if r not in neighbors]
            if invalid_receivers:
                ids = [r.id for r in invalid_receivers]
                raise ValueError(
                    f"Sender and receiver must be connected in the network; not connected receivers: {ids}"
                )

        currently_active_agents = set(self.active_agents(self._iteration))
        for r in receiver:
            if r not in currently_active_agents:
                raise ValueError(f"Receiver {r} is not active in iteration {self._iteration}")
            self._send_one(sender=sender, receiver=r, msg=msg)

    def step(self, iteration: int) -> None:
        """Set the iteration counter for the network and clear all recieved messages from agents."""
        self._iteration = iteration

        if not self._drop_unread_messages:
            return

        for agent in self.agents():
            agent._received_messages.clear()  # noqa: SLF001


class P2PNetwork(Network):
    """
    Peer-to-peer network architecture where agents communicate directly with each other.

    Args:
        graph: NetworkX graph defining the network topology. Can be a graph with arbitrary node types as long as a list
            of agents is provided via the `agents` argument; or it can be a graph with
            :class:`~decent_bench.agents.Agent` nodes, in which case the `agents` argument is optional and will be
            ignored if provided.
        agents: list of agents corresponding to the nodes in `graph` if `graph` is not a graph with
            :class:`~decent_bench.agents.Agent` nodes. The order of agents in the list should correspond to the order
            of nodes in `graph.nodes()`. This argument is ignored if `graph` is a graph with
            :class:`~decent_bench.agents.Agent` nodes.
        drop_unread_messages: whether to drop messages that are not received by the next iteration (i.e. messages that
            are "in-flight" for more than one iteration). If ``False``, messages will stay in-flight until they are
            received or replaced by a newer message from the same sender to the same receiver. After being received
            or replaced, the message is destroyed.
        message_noise: noise scheme(s) to apply to messages sent by agents in the network. Can be a single
            :class:`~decent_bench.schemes.NoiseScheme` instance to apply the same scheme to all agents, a dictionary
            mapping each agent to its scheme, or ``None`` to apply no noise to any agent.

    """

    def __init__(
        self,
        graph: AnyGraph,
        agents: Sequence[Agent] | None = None,
        *,
        drop_unread_messages: bool = True,
        message_noise: NoiseScheme | dict[Agent, NoiseScheme] | None = None,
        message_compression: CompressionScheme | dict[Agent, CompressionScheme] | None = None,
        message_drop: DropScheme | dict[Agent, DropScheme] | None = None,
    ) -> None:
        if all(isinstance(node, Agent) for node in graph.nodes()):  # pass directly to super().__init__
            super().__init__(
                graph=graph,
                drop_unread_messages=drop_unread_messages,
                message_noise=message_noise,
                message_compression=message_compression,
                message_drop=message_drop,
            )
        else:  # create AgentGraph from graph (which defines the topology) and list of agents
            if agents is None:
                raise ValueError("Provide `agents` if `graph` is not a Graph with Agent nodes")
            if len(agents) != len(graph):
                raise ValueError(f"Expected {len(graph)} agents but got {len(agents)}")
            agent_node_map = {node: agents[i] for i, node in enumerate(graph.nodes())}
            graph = nx.relabel_nodes(graph, agent_node_map)
            super().__init__(
                graph=graph,
                drop_unread_messages=drop_unread_messages,
                message_noise=message_noise,
                message_compression=message_compression,
                message_drop=message_drop,
            )
        self.W: Array | None = None

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
        degrees = self.degrees
        for i in agents:
            neighbors = self.neighbors(i)
            d_i = degrees[i]
            for j in neighbors:
                d_j = degrees[j]
                W[i, j] = 1 / (1 + max(d_i, d_j))
        for i in agents:
            W[i, i] = 1 - sum(W[i])

        self.W = iop.to_array(W, agents[0].cost.framework, agents[0].cost.device)
        return self.W

    @weights.setter
    def weights(self, value: Array) -> None:
        """
        Set custom consensus weights matrix.

        A simple way to create custom weights is to start using numpy and then
        use :func:`~decent_bench.utils.interoperability.to_array` to convert to an
        :class:`~decent_bench.utils.array.Array` object with the desired framework and device.
        For an example see :func:`~decent_bench.utils.interoperability.zeros`.

        Raises:
            ValueError: if the shape, framework, and device are incompatible with the agents' cost functions

        Note:
            If not set, the weights matrix is initialized using the Metropolis-Hastings method.
            Weights will be overwritten if framework or device differ from
            ``Agent.cost.framework`` or ``Agent.cost.device``.

        """
        if iop.shape(value) != (len(self.agents()), len(self.agents())):
            raise ValueError(f"Weights matrix must be of shape ({len(self.agents())}, {len(self.agents())})")

        framework, device = iop.framework_device_of_array(value)

        if framework != self.agents()[0].cost.framework or device != self.agents()[0].cost.device:
            raise ValueError(
                f"Weights matrix must be on the same framework and device as the agents' "
                f"cost functions ({self.agents()[0].cost.framework}, {self.agents()[0].cost.device})"
            )

        self.W = value

    @cached_property
    def adjacency(self) -> Array:
        """
        Adjacency matrix of the network.

        Use ``adjacency[i, j]`` or ``adjacency[i.id, j.id]`` to get the adjacency between agent i and j.
        """
        agents = self.agents()
        adjacency_matrix = nx.to_numpy_array(
            self.graph,
            nodelist=cast("Collection[Any]", agents),
            dtype=float,
        )  # type: ignore[call-overload]
        return iop.to_array(
            adjacency_matrix,
            agents[0].cost.framework,
            agents[0].cost.device,
        )

    def neighbors(self, agent: Agent) -> list[Agent]:
        """Alias for :meth:`~decent_bench.networks.Network.connected_agents`."""
        return super().connected_agents(agent)

    def active_neighbors(self, agent: Agent, iteration: int) -> list[Agent]:
        """Alias for :meth:`~decent_bench.networks.Network.active_connected_agents`."""
        return super().active_connected_agents(agent, iteration)

    def broadcast(self, sender: Agent, msg: Array) -> None:
        """Send to all neighbors (alias for :meth:`~decent_bench.networks.Network.send` with ``receiver=None``)."""
        self.send(sender=sender, receiver=None, msg=msg)


class FedNetwork(Network):
    """
    Federated learning network with one server node connected to all client nodes (star topology).

    Args:
        clients: list of client agents in the network.
        server: server agent in the network. If ``None``, a default server with zero cost and always active scheme will
            be created.
        drop_unread_messages: whether to drop messages that are not received by the next iteration (i.e. messages that
            are "in-flight" for more than one iteration). If ``False``, messages will stay in-flight until they are
            received or replaced by a newer message from the same sender to the same receiver. After being received
            or replaced, the message is destroyed.
        message_noise: noise scheme(s) to apply to messages sent by agents in the network. Can be a single
            :class:`~decent_bench.schemes.NoiseScheme` instance to apply the same scheme to all agents, a dictionary
            mapping each agent to its scheme, or ``None`` to apply no noise to any agent.
        message_compression: compression scheme(s) to apply to messages sent by agents in the network. Can be a single
            :class:`~decent_bench.schemes.CompressionScheme` instance to apply the same scheme to all agents,
            a dictionary mapping each agent to its scheme, or ``None`` to apply no compression to any agent.
        message_drop: drop scheme(s) to apply to messages sent by agents in the network. Can be a single
            :class:`~decent_bench.schemes.DropScheme` instance to apply the same scheme to all agents, a dictionary
            mapping each agent to its scheme, or ``None`` to apply no message drop to any agent.

    """

    def __init__(
        self,
        clients: Sequence[Agent],
        server: Agent | None = None,
        drop_unread_messages: bool = True,
        *,
        message_noise: NoiseScheme | dict[Agent, NoiseScheme] | None = None,
        message_compression: (CompressionScheme | dict[Agent, CompressionScheme] | None) = None,
        message_drop: DropScheme | dict[Agent, DropScheme] | None = None,
    ) -> None:
        if len(clients) == 0:
            raise ValueError("`clients` list must be non-empty")
        if server is None:
            # get cost info from one of the clients
            shape, framework, device = clients[0].cost.shape, clients[0].cost.framework, clients[0].cost.device
            server = Agent(
                max(c.id for c in clients) + 1,
                ZeroCost(shape, framework, device),
                AlwaysActive(),
                min(c.state_snapshot_period for c in clients),
            )
        graph = nx.star_graph([server, *list(clients)])  # create AgentGraph

        # specify the server's message schemes if not provided
        if isinstance(message_noise, dict) and server not in message_noise:
            message_noise[server] = NoNoise()
        if isinstance(message_compression, dict) and server not in message_compression:
            message_compression[server] = NoCompression()
        if isinstance(message_drop, dict) and server not in message_drop:
            message_drop[server] = NoDrops()

        super().__init__(
            graph=graph,
            drop_unread_messages=drop_unread_messages,
            message_noise=message_noise,
            message_compression=message_compression,
            message_drop=message_drop,
        )
        self._server = server

    def server(self) -> Agent:
        """Agent acting as the central server."""
        return self._server

    def coordinator(self) -> Agent:
        """Alias for :attr:`server`."""
        return self.server()

    def agents(self) -> list[Agent]:
        """Get all client agents (excludes the server/coordinator)."""
        return self._clients_cache

    @cached_property
    def _clients_cache(self) -> list[Agent]:
        """Cached list of clients; assumes the underlying graph is not mutated after construction."""
        return [agent for agent in self.graph if agent is not self.server()]

    def active_agents(self, iteration: int) -> list[Agent]:
        """Get all active client agents (excludes the server/coordinator)."""
        # Delegates to Network.active_agents(), which iterates over self.agents() (clients only for FedNetwork).
        return super().active_agents(iteration)

    def clients(self) -> list[Agent]:
        """Alias for :meth:`agents`."""
        return self.agents()

    def active_clients(self, iteration: int) -> list[Agent]:
        """Alias for :meth:`active_agents`."""
        return self.active_agents(iteration)

    def send(
        self,
        sender: Agent,
        receiver: Agent | Sequence[Agent] | None = None,
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
            if sender is self.server() and receiver is self.server():
                raise ValueError("Server-to-server communication is not supported")
            if sender is not self.server() and receiver is not self.server():
                raise ValueError("Client-to-client communication is not supported")
            super().send(sender=sender, receiver=receiver, msg=msg)
            return

        if receiver is None:
            super().send(sender=sender, receiver=receiver, msg=msg)
            return

        if sender is not self.server():
            raise ValueError("Only the server can send to multiple receivers")
        if any(r is self.server() for r in receiver):
            raise ValueError("All receivers must be clients")
        super().send(sender=sender, receiver=receiver, msg=msg)

    def broadcast(self, msg: Array) -> None:
        """Send the same message from the server to every client (synchronous FL push)."""
        self.send(sender=self.server(), receiver=None, msg=msg)
