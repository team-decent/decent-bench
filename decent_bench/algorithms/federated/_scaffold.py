from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import decent_bench.utils.interoperability as iop
from decent_bench.algorithms.utils import initial_states
from decent_bench.networks import FedNetwork
from decent_bench.schemes import ClientSelectionScheme, UniformSelection
from decent_bench.utils._tags import tags
from decent_bench.utils.types import InitialStates

from ._fed_algorithm import FedAlgorithm

if TYPE_CHECKING:
    from decent_bench.agents import Agent
    from decent_bench.utils.array import Array


_MODEL_DELTA_LABEL = "model_delta"
_CONTROL_VARIATE_DELTA_LABEL = "control_variate_delta"
_SERVER_MODEL_LABEL = "server_model"
_SERVER_CONTROL_LABEL = "server_control"


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

    Note:
        ``x0`` and ``c0`` follow the :obj:`~decent_bench.utils.types.InitialStates` convention and are resolved
        per agent during ``initialize`` via
        :func:`~decent_bench.algorithms.utils.initial_states`. For federated networks, ``c0`` dictionaries can
        include both client control variates and the server control variate. If the server entry is missing, it is
        inferred as the average of the client control variates.

    """

    iterations: int = 100
    step_size: float = 0.001
    num_local_epochs: int = 1
    server_step_size: float = 1.0
    selection_scheme: ClientSelectionScheme | None = field(
        default_factory=lambda: UniformSelection(fraction_selected_clients=1.0)
    )
    x0: InitialStates = None
    c0: InitialStates = None
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
        self.c0 = initial_states(self.c0, network)
        server = network.server()
        server_x0 = self.x0[server]
        server.initialize(
            x=server_x0,
            aux_vars={"c": self.c0[server]},
        )
        for client in network.clients():
            client_x0 = self.x0[client]
            client.initialize(
                x=client_x0,
                aux_vars={"c_i": self.c0[client]},
            )

    def step(self, network: FedNetwork, iteration: int) -> None:
        selected_clients = self.select_clients(network, iteration)
        if not selected_clients:
            return

        self.server_broadcast(network, selected_clients)
        participating_clients = self._clients_with_server_broadcast(network, selected_clients)
        if not participating_clients:
            return
        self._run_local_updates(network, participating_clients)
        self.aggregate(network, participating_clients)

    def server_broadcast(
        self,
        network: FedNetwork,
        selected_clients: Sequence["Agent"],
        label: str = "default",
    ) -> None:
        """Send the current server model and control variate under ``label``."""
        payload = iop.stack([network.server().x, network.server().aux_vars["c"]], dim=0)
        network.send(sender=network.server(), receiver=selected_clients, msg=payload, label=label)

    def _run_local_updates(self, network: FedNetwork, participating_clients: Sequence["Agent"]) -> None:
        for client in participating_clients:
            client.x, model_delta, client.aux_vars["delta_c"] = self._compute_local_update(client, network.server())
            network.send(
                sender=client,
                receiver=network.server(),
                msg=model_delta,
                label=_MODEL_DELTA_LABEL,
            )
            network.send(
                sender=client,
                receiver=network.server(),
                msg=client.aux_vars["delta_c"],
                label=_CONTROL_VARIATE_DELTA_LABEL,
            )

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

        Only current-round client deltas received by the server are aggregated.
        """
        server = network.server()
        received_model_delta_clients = set(server.messages(_MODEL_DELTA_LABEL))
        received_control_delta_clients = set(server.messages(_CONTROL_VARIATE_DELTA_LABEL))
        received_clients = [
            client
            for client in participating_clients
            if client in received_model_delta_clients and client in received_control_delta_clients
        ]
        if not received_clients:
            return
        model_deltas = [server.message(client, _MODEL_DELTA_LABEL) for client in received_clients]
        weights = [1.0] * len(received_clients)
        total_weight = float(len(received_clients))
        average_model_delta = self._weighted_average(model_deltas, weights, total_weight)
        server_x = iop.copy(server.x)
        server.x = server_x + self.server_step_size * average_model_delta

        control_deltas = [server.message(client, _CONTROL_VARIATE_DELTA_LABEL) for client in received_clients]
        average_control_delta = self._weighted_average(control_deltas, weights, total_weight)
        participation_fraction = len(received_clients) / len(network.clients())
        server.aux_vars["c"] += participation_fraction * average_control_delta
