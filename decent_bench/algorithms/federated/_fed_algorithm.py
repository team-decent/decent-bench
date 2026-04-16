from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Final, cast

import decent_bench.utils.interoperability as iop
from decent_bench.algorithms._algorithm import Algorithm
from decent_bench.algorithms.utils import infer_client_weight
from decent_bench.networks import FedNetwork
from decent_bench.schemes import ClientSelectionScheme
from decent_bench.utils.types import ClientWeights

if TYPE_CHECKING:
    from decent_bench.agents import Agent


class FedAlgorithm(Algorithm[FedNetwork]):
    r"""
    Federated algorithm - clients collaborate via a central server.

    Note:
        ``client_weights`` only affects how updates are aggregated at the server; it does not change the objective
        function being optimized (the goal is still to solve :math:`\min \sum_i f_i(x)`). To optimize a weighted
        objective :math:`\min \sum_i w_i f_i(x)`, scale each client's cost by ``w_i`` in the problem definition.

    """

    selection_scheme: ClientSelectionScheme | None = None
    _DEFAULT_SELECTION_SCHEME: Final[object] = object()
    client_weights: ClientWeights | None = None
    _DEFAULT_CLIENT_WEIGHTS: Final[object] = object()

    def cleanup_agents(self, network: FedNetwork) -> Iterable["Agent"]:
        return [network.server(), *network.clients()]

    def server_broadcast(self, network: FedNetwork, selected_clients: Sequence["Agent"]) -> None:
        """Send the current server model to the selected clients."""
        network.send(sender=network.server(), receiver=selected_clients, msg=network.server().x)

    def select_clients(
        self,
        clients: Sequence["Agent"],
        iteration: int,
        selection_scheme: ClientSelectionScheme | object | None = _DEFAULT_SELECTION_SCHEME,
    ) -> list["Agent"]:
        """
        Select participating clients from an eligible pool.

        Args:
            clients: eligible clients to select from.
            iteration: current round index.
            selection_scheme: optional override for this call. If omitted, uses ``self.selection_scheme``.
                Pass ``None`` to force selecting all clients.

        """
        if selection_scheme is self._DEFAULT_SELECTION_SCHEME:
            selection_scheme = self.selection_scheme
        if selection_scheme is None:
            return list(clients)
        scheme = cast("ClientSelectionScheme", selection_scheme)
        return scheme.select(clients, iteration)

    @classmethod
    def _weights_for_clients(
        cls,
        clients: Sequence["Agent"],
        client_weights: ClientWeights | None,
    ) -> list[float]:
        if client_weights is None:
            weights = [infer_client_weight(client) for client in clients]
        elif isinstance(client_weights, dict):
            weights = []
            for client in clients:
                if client.id not in client_weights:
                    raise ValueError(f"Missing weight for client id {client.id}")
                weights.append(float(client_weights[client.id]))
        else:
            max_id = max(client.id for client in clients)
            if len(client_weights) <= max_id:
                raise ValueError("client_weights sequence must be indexed by client id")
            weights = [float(client_weights[client.id]) for client in clients]
        if any(weight < 0 for weight in weights):
            raise ValueError("Client weights must be non-negative")
        return weights

    def aggregate(
        self,
        network: FedNetwork,
        selected_clients: Sequence["Agent"],
        client_weights: ClientWeights | object | None = _DEFAULT_CLIENT_WEIGHTS,
    ) -> None:
        """
        Aggregate client updates at the server.

        By default, this performs a weighted average of the received client models. If ``client_weights`` is not
        provided, ``self.client_weights`` is used. If weights are ``None``, they are inferred from client data size.

        Override this method for custom aggregation strategies (e.g., robust aggregation).

        Raises:
            ValueError: if the sum of client weights is non-positive.

        """
        if client_weights is self._DEFAULT_CLIENT_WEIGHTS:
            client_weights = self.client_weights

        received_clients = [client for client in selected_clients if client in network.server().messages]
        if not received_clients:
            return
        updates = [network.server().messages[client] for client in received_clients]
        weights = self._weights_for_clients(received_clients, cast("ClientWeights | None", client_weights))
        total_weight = sum(weights)
        if total_weight <= 0:
            raise ValueError("Sum of client weights must be positive")
        weighted_updates = [update * weight for update, weight in zip(updates, weights, strict=True)]
        network.server().x = iop.sum(iop.stack(weighted_updates, dim=0), dim=0) / total_weight

    def _selected_clients_for_round(self, network: FedNetwork, iteration: int) -> list["Agent"]:
        active_clients = network.active_clients()
        if not active_clients:
            return []
        return self.select_clients(active_clients, iteration)
