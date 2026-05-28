from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from uuid import UUID

import networkx as nx

from decent_bench.agents import Agent, AgentHistory
from decent_bench.costs import Cost
from decent_bench.metrics import utils
from decent_bench.networks import FedNetwork, Network, P2PNetwork


@dataclass(frozen=True, eq=False)
class AgentMetricsView:
    """Immutable view of agent that exposes useful properties for calculating metrics."""

    id: UUID
    cost: Cost
    x_history: AgentHistory
    n_x_updates: int
    n_function_calls: float
    n_gradient_calls: float
    n_hessian_calls: float
    n_proximal_calls: float
    n_sent_messages: float
    n_received_messages: float
    n_sent_messages_dropped: float
    n_times_selected: int

    @staticmethod
    def from_agent(agent: Agent) -> AgentMetricsView:
        """Create from agent."""
        return AgentMetricsView(
            id=agent.id,
            cost=agent.cost,
            x_history=agent._x_history,  # noqa: SLF001
            n_x_updates=agent._n_x_updates,  # noqa: SLF001
            n_function_calls=agent._n_function_calls,  # noqa: SLF001
            n_gradient_calls=agent._n_gradient_calls,  # noqa: SLF001
            n_hessian_calls=agent._n_hessian_calls,  # noqa: SLF001
            n_proximal_calls=agent._n_proximal_calls,  # noqa: SLF001
            n_sent_messages=agent._n_sent_messages,  # noqa: SLF001
            n_received_messages=agent._n_received_messages,  # noqa: SLF001
            n_sent_messages_dropped=agent._n_sent_messages_dropped,  # noqa: SLF001
            n_times_selected=agent._n_times_selected,  # noqa: SLF001
        )


class NetworkType(Enum):
    """Supported network types for metric views."""

    P2P = "p2p"
    FEDERATED = "federated"


@dataclass(frozen=True)
class NetworkMetricsView:
    """
    Immutable view of a network that exposes useful properties for calculating metrics.

    The underlying data structure is a frozen ``nx.Graph`` whose nodes are ``AgentMetricsView`` objects.
    The object is created using ``from_network`` passing a ``FedNetwork`` or ``P2PNetwork``.

    Available methods are:
    - ``agents()`` and ``connected_agents(agent)``
    - Fed-only: ``clients()``, ``server()``, and ``coordinator()``
    - P2P-only: ``neighbors(agent)``
    """

    graph: nx.Graph[AgentMetricsView]
    network_type: NetworkType
    _server: AgentMetricsView | None = None

    @staticmethod
    def from_network(network: Network) -> NetworkMetricsView:
        """Create a network metrics view from a network."""
        snapshot_agents = network.snapshot_agents()
        agent_views = [AgentMetricsView.from_agent(agent) for agent in snapshot_agents]
        agent_map = dict(zip(snapshot_agents, agent_views, strict=True))

        relabeled_graph = nx.relabel_nodes(network.graph, agent_map, copy=True)
        frozen_graph = nx.freeze(relabeled_graph.copy())

        if isinstance(network, FedNetwork):
            server_view = agent_map[network.server()]
            return NetworkMetricsView(
                graph=frozen_graph,
                network_type=NetworkType.FEDERATED,
                _server=server_view,
            )
        if isinstance(network, P2PNetwork):
            return NetworkMetricsView(graph=frozen_graph, network_type=NetworkType.P2P)

        raise ValueError(f"Unsupported network type: {type(network)!r}")

    def agents(self) -> list[AgentMetricsView]:
        """Return agents exposed by network semantics (clients for federated, all nodes for P2P)."""
        if self.network_type is NetworkType.FEDERATED:
            return [agent for agent in self.graph.nodes if agent is not self._server]
        return list(self.graph.nodes)

    def clients(self) -> list[AgentMetricsView]:
        """Return clients in a federated network (alias of agents())."""
        if self.network_type is not NetworkType.FEDERATED:
            raise ValueError("clients() is only available for federated networks")
        return self.agents()

    def server(self) -> AgentMetricsView:
        """Return the server node in a federated network."""
        if self.network_type is not NetworkType.FEDERATED or self._server is None:
            raise ValueError("server() is only available for federated networks")
        return self._server

    def coordinator(self) -> AgentMetricsView:
        """Alias for server()."""
        return self.server()

    def connected_agents(self, agent: AgentMetricsView) -> list[AgentMetricsView]:
        """Return agents in the network connected to an agent."""
        if agent not in self.graph:
            raise ValueError("agent is not in the network")
        return list(self.graph.neighbors(agent))

    def neighbors(self, agent: AgentMetricsView) -> list[AgentMetricsView]:
        """Return neighbors in a peer-to-peer network."""
        if self.network_type is not NetworkType.P2P:
            raise ValueError("neighbors() is only available for p2p networks")
        return self.connected_agents(agent)

    @property
    def iterations(self) -> list[int]:
        """List of iterations reached by any agent (plus server) in the network."""
        return utils.all_sorted_iterations(self.graph.nodes)
