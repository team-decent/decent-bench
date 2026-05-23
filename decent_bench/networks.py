from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Sequence
from copy import deepcopy
from functools import cached_property
from typing import Any

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


class Network(ABC):  # noqa: B024
    """
    Base network object defining communication logic shared by all network types.

    Args:
        graph: NetworkX graph defining the network topology, with :class:`~decent_bench.agents.Agent` nodes.
        message_noise: noise scheme(s) to apply to messages sent by agents in the network. Can be a single
            :class:`~decent_bench.schemes.NoiseScheme` instance when all agents use the same kind of noise scheme, a
            dictionary mapping each agent to its scheme when agents use different schemes, or ``None`` to apply no
            noise to any agent.
        message_compression: compression scheme(s) to apply to messages sent by agents in the network. Can be a single
            :class:`~decent_bench.schemes.CompressionScheme` instance when all agents use the same kind of compression
            scheme, a dictionary mapping each agent to its scheme when agents use different schemes, or ``None`` to
            apply no compression to any agent.
        message_drop: drop scheme(s) to apply to messages sent by agents in the network. Can be a single
            :class:`~decent_bench.schemes.DropScheme` instance when all agents use the same kind of drop scheme, a
            dictionary mapping each agent to its scheme when agents use different schemes, or ``None`` to apply no
            message drop to any agent.

    Raises:
        ValueError: if the graph is not connected, if it is directed, if it is a multi-graph, if its nodes are
            not of type :class:`~decent_bench.agents.Agent`, if any two agents have incompatible costs, or if
            any agent is already assigned to another network (i.e. it has index != -1).

    """

    def __init__(
        self,
        graph: nx.Graph[Agent],
        message_noise: NoiseScheme | dict[Agent, NoiseScheme] | None = None,
        message_compression: CompressionScheme | dict[Agent, CompressionScheme] | None = None,
        message_drop: DropScheme | dict[Agent, DropScheme] | None = None,
    ) -> None:
        # check that graph is connected and not a multi-graph
        if graph.is_multigraph():
            raise NotImplementedError("Support for multi-graphs is not available")
        if graph.is_directed():
            raise ValueError("Directed graphs are not supported; please provide an undirected graph")
        if not nx.is_connected(graph):
            raise ValueError("The graph needs to be connected")
        if any(not isinstance(node, Agent) for node in graph):
            raise ValueError("The graph nodes must be `Agent` objects")
        self._validate_agent_cost_compatibility(graph)
        for idx, agent in enumerate(graph.nodes()):  # assign agent index within the network
            agent.index = idx

        self._graph = graph
        self._message_noise = self._initialize_message_schemes(message_noise, "noise", NoiseScheme, NoNoise)
        self._message_compression = self._initialize_message_schemes(
            message_compression, "compression", CompressionScheme, NoCompression
        )
        self._message_drop = self._initialize_message_schemes(message_drop, "drop", DropScheme, NoDrops)
        self._active_agents_cache: list[Agent] | None = None
        self._active_connected_agents_cache: dict[Agent, list[Agent]] = {}
        self._iteration = 0  # Current iteration, updated by the algorithm

    @staticmethod
    def _validate_agent_cost_compatibility(graph: nx.Graph[Agent]) -> None:
        """
        Validate that all agents' costs share the same shape, framework, and device.

        Raises:
            ValueError: If agents in the graph have mismatching cost shape, framework, or device.

        """
        agents = list(graph.nodes())
        if len(agents) <= 1:
            return

        first_cost = agents[0].cost
        first_signature = (first_cost.shape, first_cost.framework, first_cost.device)
        mismatches: list[str] = []
        for agent in agents[1:]:
            signature = (agent.cost.shape, agent.cost.framework, agent.cost.device)
            if signature != first_signature:
                mismatches.append(
                    f"agent {agent.id}: shape={agent.cost.shape}, framework={agent.cost.framework}, "
                    f"device={agent.cost.device}"
                )

        if mismatches:
            raise ValueError(
                "All agents in a network must have costs with the same shape, framework, and device. "
                f"Expected shape={first_cost.shape}, framework={first_cost.framework}, "
                f"device={first_cost.device}; mismatches: {'; '.join(mismatches)}"
            )

    @staticmethod
    def _validate_agent_ids(agents: Sequence[Agent]) -> None:
        """
        Validate that all agents have distinct ids and were not assigned to a network.

        This util checks that all agents have distinct ``Agent.id``. Distinct ids are mandatory since agents are
        hashed by their id (necessary during un/pickling operations). The util is meant to be used in subclasses
        on raw Agent lists passed by users. NetworkX collapses two agents with the same id, so it is necessary to
        run this validation before Graph[Agent] is constructed.

        Additionally, it checks that no agent was previously assigned to another network, by checking that its
        index is -1.

        Raises:
            ValueError: If any two agents have the same id or were already assigned to a network.

        """
        agent_ids = [agent.id for agent in agents]
        if len(agent_ids) != len(set(agent_ids)):
            raise ValueError("Agent IDs must be unique")

        if any(agent.index != -1 for agent in agents):
            raise ValueError("Agents can only be assigned to one network at a time")

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
        - a single `scheme_class` instance: apply an independent copy to every agent
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
        if isinstance(scheme, scheme_class):  # one scheme, copy for all agents
            return {agent: deepcopy(scheme) for agent in self.graph}
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
    def graph(self) -> nx.Graph[Agent]:
        """Underlying NetworkX graph; mutating it will change the network."""
        return self._graph

    @property
    def G(self) -> nx.Graph[Agent]:  # noqa: N802
        """Alias for the underlying graph."""
        return self.graph

    def agents(self) -> list[Agent]:
        """
        Get all agents in the network.

        Warning:
            This includes agents that may be inactive in the current iteration.
            It is generally recommended to use :meth:`active_agents` instead, which returns only active agents,
            unless you have a specific reason to access inactive agents or need to access all agents regardless
            of activation status.

        """
        return self._agents_cache

    def snapshot_agents(self) -> list[Agent]:
        """Get agents whose state should be snapshotted during algorithm execution."""
        return list(self.graph.nodes())

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

    def active_agents(self) -> list[Agent]:
        """
        Get all *active* agents.

        Whether an :class:`~decent_bench.agents.Agent` is active or not at a given time is defined by its
        :class:`~decent_bench.schemes.AgentActivationScheme`.
        """
        if self._active_agents_cache is None:
            # Cache the active agents for each iterations in case the same iteration is queried multiple times
            # so that we preserve the activated agents for the iteration even if the underlying activation schemes
            # are non-deterministic.
            self._active_agents_cache = [
                a
                for a in self.agents()
                if a._activation.is_active(self._iteration)  # noqa: SLF001
            ]
        return self._active_agents_cache

    def connected_agents(self, agent: Agent) -> list[Agent]:
        """
        Agents directly connected to ``agent`` in the underlying graph.

        Warning:
            This includes agents that may be inactive in the current iteration.
            It is generally recommended to use :meth:`active_connected_agents` instead, which returns only active
            agents, unless you have a specific reason to access inactive agents or need to access all connected agents
            regardless of activation status.

        """
        return list(self.graph.neighbors(agent))

    def active_connected_agents(self, agent: Agent) -> list[Agent]:
        """Agents directly connected to ``agent`` and are active at the given iteration."""
        if agent not in self._active_connected_agents_cache:
            active_agents = set(self.active_agents())
            self._active_connected_agents_cache[agent] = [a for a in self.connected_agents(agent) if a in active_agents]
        return self._active_connected_agents_cache[agent]

    def _allowed_receivers(self, sender: Agent) -> set[Agent]:
        """Get the set of agents that the sender is allowed to send messages to (i.e. connected and active)."""
        return set(self.active_connected_agents(sender))

    def send(
        self,
        sender: Agent,
        receiver: Agent | Sequence[Agent] | None = None,
        msg: Array | None = None,
    ) -> None:
        """
        Send message to one or more agents.

        The message may be compressed, distorted by noise, and/or dropped depending on the network's
        :class:`~decent_bench.schemes.CompressionScheme`,
        :class:`~decent_bench.schemes.NoiseScheme`,
        and :class:`~decent_bench.schemes.DropScheme`. The potentially compressed and noisy message (if not dropped)
        is then instantaneously available to each receiver.

        Args:
            sender: sender agent
            receiver: receiver agent, sequence of receiver agents, or ``None`` to broadcast to connected agents.
            msg: array message to send

        Raises:
            ValueError: if ``msg`` is not provided, if agents are not part of the network, or if sender/receiver are not
                connected and active.

        """
        if msg is None:
            raise ValueError("msg must be provided")

        if sender not in self.graph:
            raise ValueError("Sender must be an agent in the network")

        if receiver is None:
            receiver = self.active_connected_agents(sender)
        elif isinstance(receiver, Agent):
            if receiver not in self.connected_agents(sender):
                raise ValueError("Sender and receiver must be connected in the network")
            receiver = [receiver]

        # raise for unconnected/inactive
        unavailable_missing_receivers = set(receiver) - self._allowed_receivers(sender)
        if unavailable_missing_receivers:
            raise ValueError(
                f"Receivers {unavailable_missing_receivers} are not active or not connected to {sender} "
                f"in iteration {self._iteration}"
            )

        counter_increment = self._message_compression[sender].compressed_msg_size(msg) / sender.cost.size
        sender._n_sent_messages += counter_increment * len(receiver)  # noqa: SLF001
        framework, device = iop.framework_device_of_array(msg)  # remove after iop refactor

        # select confirmed receivers (message is not dropped)
        confirmed_receivers = [r for r in receiver if not self._message_drop[sender].should_drop()]
        sender._n_sent_messages_dropped += counter_increment * (len(receiver) - len(confirmed_receivers))  # noqa: SLF001
        # compress the message
        msg = self._message_compression[sender].compress(iop.copy(msg))
        # generate noise
        noise = self._message_noise[sender].make_noise((len(confirmed_receivers), *iop.shape(msg)), framework, device)
        # transmit messages
        for i, r in enumerate(confirmed_receivers):
            r._received_messages[sender] = msg if noise is None else msg + noise[i]  # noqa: SLF001
            r._n_received_messages += counter_increment  # noqa: SLF001

    def _step(self, iteration: int) -> None:
        """Set the iteration counter for the network and clear all received messages from agents."""
        self._iteration = iteration

        # Clear caches
        self._active_agents_cache = None
        self._active_connected_agents_cache = {}

        self._clear_received_messages()

    def _clear_received_messages(
        self,
        receivers: Sequence[Agent] | None = None,
        senders: Sequence[Agent] | None = None,
    ) -> None:
        """Clear received messages, optionally scoped to specific receivers and senders."""
        # Use the _agents_cache to avoid overridden agents() in subclasses like FedNetwork
        receivers = self._agents_cache if receivers is None else receivers
        for receiver in receivers:
            if senders is None:
                receiver._received_messages.clear()  # noqa: SLF001
            else:
                for sender in senders:
                    receiver._received_messages.pop(sender, None)  # noqa: SLF001


class P2PNetwork(Network):
    """
    Peer-to-peer network architecture where agents communicate directly with each other.

    Args:
        graph: NetworkX graph defining the network topology.
        agents: list of agents corresponding to the nodes in ``graph``. The agents in the list are assigned
            in order to each node of the graph. Agents must have unique ids.
        message_noise: noise scheme(s) to apply to messages sent by agents in the network. Can be a single
            :class:`~decent_bench.schemes.NoiseScheme` instance to apply the same scheme to all agents, a dictionary
            mapping each agent to its scheme, or ``None`` to apply no noise to any agent.
        message_compression: compression scheme(s) to apply to messages sent by agents in the network. Can be a single
            :class:`~decent_bench.schemes.CompressionScheme` instance to apply the same scheme to all agents, a
            dictionary mapping each agent to its scheme, or ``None`` to apply no compression to any agent.
        message_drop: drop scheme(s) to apply to messages sent by agents in the network. Can be a single
            :class:`~decent_bench.schemes.DropScheme` instance to apply the same scheme to all agents, a dictionary
            mapping each agent to its scheme, or ``None`` to apply no message drop to any agent.

    Raises:
        ValueError: if length of ``agents`` doesn't match the number of nodes in ``graph``, or if any two agents
            have the same ids.

    """

    def __init__(
        self,
        graph: nx.Graph[Any],
        agents: Sequence[Agent],
        *,
        message_noise: NoiseScheme | dict[Agent, NoiseScheme] | None = None,
        message_compression: CompressionScheme | dict[Agent, CompressionScheme] | None = None,
        message_drop: DropScheme | dict[Agent, DropScheme] | None = None,
    ) -> None:
        if len(agents) != len(graph):
            raise ValueError(f"Expected {len(graph)} agents but got {len(agents)}")
        self._validate_agent_ids(agents)

        agent_node_map = {node: agents[i] for i, node in enumerate(graph.nodes())}
        graph = nx.relabel_nodes(graph, agent_node_map)
        super().__init__(
            graph=graph,
            message_noise=message_noise,
            message_compression=message_compression,
            message_drop=message_drop,
        )
        self.W: Array | None = None

    @property
    def weights(self) -> Array:
        """
        Symmetric, doubly stochastic matrix for consensus weights. Initialized using the Metropolis-Hastings method.

        Use ``weights[i, j]`` or ``weights[i.index, j.index]`` to get the weight between agent i and j.
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

        Use ``adjacency[i, j]`` or ``adjacency[i.index, j.index]`` to get the adjacency between agent i and j.
        """
        agents = self.agents()
        adjacency_matrix = nx.to_numpy_array(
            self.graph,
            nodelist=agents,
            dtype=np.dtype(float),
        )
        return iop.to_array(
            adjacency_matrix,
            agents[0].cost.framework,
            agents[0].cost.device,
        )

    def neighbors(self, agent: Agent) -> list[Agent]:
        """Alias for :meth:`~decent_bench.networks.Network.connected_agents`."""
        return super().connected_agents(agent)

    def active_neighbors(self, agent: Agent) -> list[Agent]:
        """Alias for :meth:`~decent_bench.networks.Network.active_connected_agents`."""
        return super().active_connected_agents(agent)

    def broadcast(self, sender: Agent, msg: Array) -> None:
        """Send to all neighbors (alias for :meth:`~decent_bench.networks.Network.send` with ``receiver=None``)."""
        self.send(sender=sender, receiver=None, msg=msg)


class FedNetwork(Network):
    """
    Federated learning network with one server node connected to all client nodes (star topology).

    Args:
        clients: list of client agents in the network.
        server: server agent in the network. If ``None``, a default server with zero cost and always active scheme will
            be created. Custom servers must use :class:`~decent_bench.schemes.AlwaysActive`.
        message_noise: noise scheme(s) to apply to messages sent by agents in the network. Can be a single
            :class:`~decent_bench.schemes.NoiseScheme` instance when all agents use the same kind of noise scheme, a
            dictionary mapping each agent to its scheme when agents use different schemes, or ``None`` to apply no
            noise to any agent.
        message_compression: compression scheme(s) to apply to messages sent by agents in the network. Can be a single
            :class:`~decent_bench.schemes.CompressionScheme` instance when all agents use the same kind of compression
            scheme, a dictionary mapping each agent to its scheme when agents use different schemes, or ``None`` to
            apply no compression to any agent.
        message_drop: drop scheme(s) to apply to messages sent by agents in the network. Can be a single
            :class:`~decent_bench.schemes.DropScheme` instance when all agents use the same kind of drop scheme, a
            dictionary mapping each agent to its scheme when agents use different schemes, or ``None`` to apply no
            message drop to any agent.

    Raises:
        ValueError: if ``clients`` is empty or if a custom ``server`` does not use
            :class:`~decent_bench.schemes.AlwaysActive`.

    """

    def __init__(
        self,
        clients: Sequence[Agent],
        server: Agent | None = None,
        *,
        message_noise: NoiseScheme | dict[Agent, NoiseScheme] | None = None,
        message_compression: CompressionScheme | dict[Agent, CompressionScheme] | None = None,
        message_drop: DropScheme | dict[Agent, DropScheme] | None = None,
    ) -> None:
        if len(clients) == 0:
            raise ValueError("`clients` list must be non-empty")
        if server is None:
            # get cost info from one of the clients
            shape, framework, device = clients[0].cost.shape, clients[0].cost.framework, clients[0].cost.device
            server = Agent(
                ZeroCost(shape, framework, device),
                AlwaysActive(),
                min(c.state_snapshot_period for c in clients),
            )
        elif not isinstance(server._activation, AlwaysActive):  # noqa: SLF001
            raise ValueError("FedNetwork server must use AlwaysActive activation")
        for client in clients:
            client._is_server = False  # noqa: SLF001
        server._is_server = True  # noqa: SLF001

        nodes = [server, *list(clients)]
        self._validate_agent_ids(nodes)
        graph = nx.star_graph(nodes)  # create Graph of Agents

        # specify the server's message schemes if not provided
        if isinstance(message_noise, dict) and server not in message_noise:
            message_noise[server] = NoNoise()
        if isinstance(message_compression, dict) and server not in message_compression:
            message_compression[server] = NoCompression()
        if isinstance(message_drop, dict) and server not in message_drop:
            message_drop[server] = NoDrops()

        super().__init__(
            graph=graph,
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
        """
        Get all client agents (excludes the server/coordinator).

        Warning:
            This includes agents that may be inactive in the current iteration.
            It is generally recommended to use :meth:`active_agents` instead, which returns only active agents,
            unless you have a specific reason to access inactive agents or need to access all agents regardless
            of activation status.

        """
        return self._clients_cache

    @cached_property
    def _clients_cache(self) -> list[Agent]:
        """Cached list of clients; assumes the underlying graph is not mutated after construction."""
        return [agent for agent in self.graph if agent is not self.server()]

    def active_agents(self) -> list[Agent]:
        """
        Get all *active* client agents (excludes the server/coordinator).

        Whether an :class:`~decent_bench.agents.Agent` is active or not at a given time is defined by its
        :class:`~decent_bench.schemes.AgentActivationScheme`.
        """
        # Delegates to Network.active_agents(), which iterates over self.agents() (clients only for FedNetwork).
        return super().active_agents()

    def clients(self) -> list[Agent]:
        """Alias for :meth:`agents`."""
        return self.agents()

    def active_clients(self) -> list[Agent]:
        """Alias for :meth:`active_agents`."""
        return self.active_agents()

    def _allowed_receivers(self, sender: Agent) -> set[Agent]:
        """Get the set of agents that the sender is allowed to send messages to (i.e. connected and active)."""
        if sender is not self.server():
            return {self.server()} if self.server()._activation.is_active(self._iteration) else set()  # noqa: SLF001
        return set(self.active_connected_agents(sender))

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
            if sender is not self.server():
                super().send(sender=sender, receiver=self.server(), msg=msg)
                return
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
