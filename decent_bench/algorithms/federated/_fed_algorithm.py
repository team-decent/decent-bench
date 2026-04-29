from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import decent_bench.utils.interoperability as iop
from decent_bench.algorithms._algorithm import Algorithm
from decent_bench.networks import FedNetwork
from decent_bench.schemes import ClientSelectionScheme

if TYPE_CHECKING:
    from decent_bench.agents import Agent
    from decent_bench.utils.array import Array


class FedAlgorithm(Algorithm[FedNetwork]):
    """Federated algorithm - clients collaborate via a central server."""

    selection_scheme: ClientSelectionScheme | None = None

    def cleanup_agents(self, network: FedNetwork) -> Iterable["Agent"]:
        return [network.server(), *network.clients()]

    def server_broadcast(self, network: FedNetwork, selected_clients: Sequence["Agent"]) -> None:
        """Send the current server model to the selected clients."""
        network.send(sender=network.server(), receiver=selected_clients, msg=network.server().x)

    def _clients_with_server_broadcast(self, network: FedNetwork, selected_clients: Sequence["Agent"]) -> list["Agent"]:
        """Return the selected clients that actually received the current server broadcast."""
        return [client for client in selected_clients if network.server() in client.messages]

    def _clear_buffered_server_messages(self, network: FedNetwork, participating_clients: Sequence["Agent"]) -> None:
        """
        Remove stale client-to-server messages for the current participants.

        This ensures the server aggregates only updates actually received from those clients in the current round.
        The clean-up is needed when :class:`~decent_bench.networks.Network` uses ``buffer_messages=True``, since old
        client uploads would otherwise remain stored at the server and could be mistaken for fresh round updates.
        """
        for client in participating_clients:
            network.server()._received_messages.pop(client, None)  # noqa: SLF001

    @staticmethod
    def _get_server_broadcast(client: "Agent", server: "Agent") -> "Array":
        """
        Return the current server broadcast received by the client.

        Raises:
            ValueError: if the client did not receive the current server broadcast.

        """
        if server not in client.messages:
            raise ValueError("Client did not receive the current server broadcast")
        return iop.copy(client.messages[server])

    @staticmethod
    def _weighted_average(values: Sequence["Array"], weights: Sequence[float], total_weight: float) -> "Array":
        """Compute a weighted average of same-shaped arrays."""
        weighted_values = [value * weight for value, weight in zip(values, weights, strict=True)]
        return iop.sum(iop.stack(weighted_values, dim=0), dim=0) / total_weight

    def _selected_clients_for_round(self, network: FedNetwork, iteration: int) -> list["Agent"]:
        """
        Select participating clients for the current round from the currently active clients.

        If ``self.selection_scheme`` is ``None``, all active clients are selected.
        """
        active_clients = network.active_clients()
        if not active_clients:
            return []
        if self.selection_scheme is None:
            return list(active_clients)
        return self.selection_scheme.select(active_clients, iteration)

    def aggregate(
        self,
        network: FedNetwork,
        participating_clients: Sequence["Agent"],
    ) -> None:
        """
        Aggregate client model uploads at the server using uniform averaging.

        This default federated aggregation assumes clients upload final local model states.

        When used with :class:`~decent_bench.networks.Network` ``buffer_messages=True``, this method assumes the
        caller has already removed stale buffered client-to-server messages for the participating clients, so only
        current-round updates are aggregated.
        """
        received_clients = [client for client in participating_clients if client in network.server().messages]
        if not received_clients:
            return
        updates = [network.server().messages[client] for client in received_clients]
        weights = [1.0] * len(received_clients)
        total_weight = float(len(received_clients))
        network.server().x = self._weighted_average(updates, weights, total_weight)
