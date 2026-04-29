from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import decent_bench.utils.interoperability as iop
from decent_bench.algorithms.utils import initial_states
from decent_bench.networks import FedNetwork
from decent_bench.schemes import ClientSelectionScheme, UniformClientSelection
from decent_bench.utils._tags import tags
from decent_bench.utils.types import InitialStates

from ._fed_algorithm import FedAlgorithm

if TYPE_CHECKING:
    from decent_bench.agents import Agent
    from decent_bench.utils.array import Array


@tags("federated")
@dataclass(eq=False)
class Scaffold(FedAlgorithm):
    r"""
    SCAFFOLD with client/server control variates for variance-reduced local training :footcite:p:`Alg_SCAFFOLD`.

    When the server control variate :math:`\mathbf{c}` and all client control variates :math:`\mathbf{c}_i`
    are zero, the local updates reduce to :class:`FedAvg <decent_bench.algorithms.federated.FedAvg>`.

    Here, :math:`\eta_l` is the local client step size, :math:`\eta_g` is the global server step size,
    :math:`S` is the set of selected clients, and :math:`|S|` its size.

    Selected clients perform local steps with the correction

    .. math::
        \mathbf{y}_{i}^{(t+1)} = \mathbf{y}_{i}^{(t)} - \eta_l
        \left(\nabla f_i(\mathbf{y}_{i}^{(t)}) - \mathbf{c}_i + \mathbf{c}\right)

    and update their control variates using the practical SCAFFOLD rule, where :math:`K` is the number of
    local steps and :math:`\mathbf{y}_i` is the final local model after those :math:`K` steps:

    .. math::
        \mathbf{c}_i^+ = \mathbf{c}_i - \mathbf{c}
        + \frac{1}{K \eta_l} (\mathbf{x} - \mathbf{y}_i).

    The server aggregates the model and control-variate deltas as

    .. math::
        \Delta \mathbf{x} = \frac{1}{|S|} \sum_{i \in S} (\mathbf{y}_i - \mathbf{x})

    .. math::
        \Delta \mathbf{c} = \frac{1}{|S|} \sum_{i \in S} (\mathbf{c}_i^+ - \mathbf{c}_i)

    and then applies

    .. math::
        \mathbf{x} \leftarrow \mathbf{x} + \eta_g \Delta \mathbf{x}, \qquad
        \mathbf{c} \leftarrow \mathbf{c} + \frac{|S|}{N} \Delta \mathbf{c},

    where both aggregated deltas are averaged uniformly over the selected clients.

    .. footbibliography::
    """

    iterations: int = 100
    step_size: float = 0.001
    num_local_epochs: int = 1
    server_step_size: float = 1.0
    selection_scheme: ClientSelectionScheme | None = field(
        default_factory=lambda: UniformClientSelection(client_fraction=1.0)
    )
    x0: InitialStates = None
    name: str = "Scaffold"

    def __post_init__(self) -> None:
        """
        Validate hyperparameters.

        Raises:
            ValueError: if hyperparameters are invalid.

        """
        if self.step_size <= 0:
            raise ValueError("`step_size` must be positive")
        if self.num_local_epochs <= 0:
            raise ValueError("`num_local_epochs` must be positive")
        if self.server_step_size <= 0:
            raise ValueError("`server_step_size` must be positive")

    def initialize(self, network: FedNetwork) -> None:
        self.x0 = initial_states(self.x0, network)
        server = network.server()
        server_x0 = self.x0[server]
        server.initialize(
            x=server_x0,
            aux_vars={"c": iop.zeros_like(server_x0)},
        )
        for client in network.clients():
            client_x0 = self.x0[client]
            client.initialize(
                x=client_x0,
                aux_vars={"c_i": iop.zeros_like(client_x0)},
            )

    def step(self, network: FedNetwork, iteration: int) -> None:
        selected_clients = self._selected_clients_for_round(network, iteration)
        if not selected_clients:
            return

        self.server_broadcast(network, selected_clients)
        participating_clients = self._clients_with_server_broadcast(network, selected_clients)
        if not participating_clients:
            return
        self._clear_buffered_server_messages(network, participating_clients)
        self._run_local_updates(network, participating_clients)
        self.aggregate(network, participating_clients)

    def server_broadcast(self, network: FedNetwork, selected_clients: Sequence["Agent"]) -> None:
        """Send the current server model and server control variate to the selected clients."""
        payload = iop.stack([network.server().x, network.server().aux_vars["c"]], dim=0)
        network.send(sender=network.server(), receiver=selected_clients, msg=payload)

    def _run_local_updates(self, network: FedNetwork, participating_clients: Sequence["Agent"]) -> None:
        for client in participating_clients:
            client.x, model_delta, client.aux_vars["delta_c"] = self._compute_local_update(client, network.server())
            payload = iop.stack([model_delta, client.aux_vars["delta_c"]], dim=0)
            network.send(sender=client, receiver=network.server(), msg=payload)

    def _compute_local_update(self, client: "Agent", server: "Agent") -> tuple["Array", "Array", "Array"]:
        """
        Run local SCAFFOLD steps using the batching semantics of ``client.cost.gradient``.

        Costs that preserve the empirical-risk abstraction default ``gradient`` to ``indices="batch"``, so SCAFFOLD
        performs mini-batch local updates automatically. Generic costs keep their usual full-gradient behavior.
        """
        server_broadcast = self._get_server_broadcast(client, server)
        reference_x = iop.copy(server_broadcast[0])
        local_x = iop.copy(reference_x)
        client_control = client.aux_vars["c_i"]
        server_control = iop.copy(server_broadcast[1])

        for _ in range(self.num_local_epochs):
            grad = client.cost.gradient(local_x)
            local_x -= self.step_size * (grad - client_control + server_control)

        new_client_control = (
            client_control - server_control + ((reference_x - local_x) / (self.num_local_epochs * self.step_size))
        )
        model_delta = local_x - reference_x
        control_variate_delta = new_client_control - client_control
        client.aux_vars["c_i"] = new_client_control
        return local_x, model_delta, control_variate_delta

    def aggregate(
        self,
        network: FedNetwork,
        participating_clients: Sequence["Agent"],
    ) -> None:
        """
        Aggregate received SCAFFOLD client deltas using uniform averaging.

        When used with :class:`~decent_bench.networks.Network` ``buffer_messages=True``, this method assumes the
        caller has already removed stale buffered client-to-server messages for the participating clients, so only
        current-round updates are aggregated.
        """
        server = network.server()
        received_clients = [client for client in participating_clients if client in server.messages]
        if not received_clients:
            return

        uploads = [server.messages[client] for client in received_clients]
        model_deltas = [upload[0] for upload in uploads]
        weights = [1.0] * len(received_clients)
        total_weight = float(len(received_clients))
        average_model_delta = self._weighted_average(model_deltas, weights, total_weight)
        server_x = iop.copy(server.x)
        server.x = server_x + self.server_step_size * average_model_delta

        control_deltas = [upload[1] for upload in uploads]
        average_control_delta = self._weighted_average(control_deltas, weights, total_weight)
        participation_fraction = len(received_clients) / len(network.clients())
        server.aux_vars["c"] += participation_fraction * average_control_delta
