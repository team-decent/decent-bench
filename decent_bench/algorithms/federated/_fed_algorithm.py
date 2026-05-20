from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import decent_bench.utils.interoperability as iop
from decent_bench.algorithms._algorithm import Algorithm
from decent_bench.networks import FedNetwork
from decent_bench.schemes import ClientSelectionScheme
from decent_bench.utils.types import LocalSteps

if TYPE_CHECKING:
    from decent_bench.agents import Agent
    from decent_bench.utils.array import Array


class FedAlgorithm(Algorithm[FedNetwork]):
    """Federated algorithm - clients collaborate via a central server."""

    selection_scheme: ClientSelectionScheme | None = None
    num_local_steps: LocalSteps = 1

    def cleanup_agents(self, network: FedNetwork) -> Iterable["Agent"]:
        return [network.server(), *network.clients()]

    def _validate_num_local_steps(self) -> None:
        """
        Validate homogeneous or per-client local step counts.

        Raises:
            TypeError: if ``num_local_steps`` is not an integer or client mapping.
            ValueError: if ``num_local_steps`` contains non-positive step counts.

        """
        if isinstance(self.num_local_steps, int):
            if self.num_local_steps <= 0:
                raise ValueError("`num_local_steps` must be positive")
            return
        if isinstance(self.num_local_steps, dict):
            for step in self.num_local_steps.values():
                if not isinstance(step, int):
                    raise TypeError("`num_local_steps` mapping values must be integers")
                if step <= 0:
                    raise ValueError("`num_local_steps` must have positive values")
            return
        raise TypeError("`num_local_steps` must be an int or a mapping from Agent to integer values")

    def _settle_num_local_steps(self, network: FedNetwork) -> dict["Agent", int]:
        """
        Resolve homogeneous or per-client local step counts for the network clients.

        Raises:
            ValueError: if a per-client mapping is missing a network client.

        """
        clients = network.clients()
        if isinstance(self.num_local_steps, int):
            return dict.fromkeys(clients, self.num_local_steps)
        missing_clients = [client for client in clients if client not in self.num_local_steps]
        if missing_clients:
            raise ValueError(
                "`num_local_steps` mapping must provide a value for every network client; "
                f"missing clients: {missing_clients}"
            )
        return {client: self.num_local_steps[client] for client in clients}

    def selected_clients_for_round(self, network: FedNetwork, iteration: int) -> list["Agent"]:
        """
        Select participating clients for the current round and record participation.

        Use this method once per federated round when selecting participating clients in the round. It applies
        ``self.selection_scheme`` to the currently active clients and updates the selection counters used by
        participation metrics. Calling it more than once in the same round will increment the selected-client counters
        more than once.

        If ``self.selection_scheme`` is ``None``, all active clients are selected.
        """
        active_clients = network.active_clients()
        if not active_clients:
            return []
        if self.selection_scheme is None:
            selected_clients = list(active_clients)
        else:
            selected_clients = self.selection_scheme.select(active_clients, iteration)
        self._record_selected_clients(selected_clients)
        return selected_clients

    @staticmethod
    def _record_selected_clients(selected_clients: Sequence["Agent"]) -> None:
        """Record clients selected by a federated round."""
        for client in selected_clients:
            client._n_times_selected = getattr(client, "_n_times_selected", 0) + 1  # noqa: SLF001

    def server_broadcast(self, network: FedNetwork, selected_clients: Sequence["Agent"]) -> None:
        """Send the current server model to the selected clients."""
        network.send(sender=network.server(), receiver=selected_clients, msg=network.server().x)

    def _clients_with_server_broadcast(self, network: FedNetwork, selected_clients: Sequence["Agent"]) -> list["Agent"]:
        """Return selected clients that received the current broadcast sent by :meth:`server_broadcast`."""
        return [client for client in selected_clients if network.server() in client.messages]

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

    def aggregate(
        self,
        network: FedNetwork,
        participating_clients: Sequence["Agent"],
    ) -> None:
        """
        Aggregate client model uploads at the server using uniform averaging.

        This default federated aggregation assumes clients upload final local model states.
        """
        received_clients = [client for client in participating_clients if client in network.server().messages]
        if not received_clients:
            return
        updates = [network.server().messages[client] for client in received_clients]
        weights = [1.0] * len(received_clients)
        total_weight = float(len(received_clients))
        network.server().x = self._weighted_average(updates, weights, total_weight)
