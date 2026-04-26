from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import decent_bench.utils.interoperability as iop
from decent_bench.algorithms.utils import initial_states
from decent_bench.networks import FedNetwork
from decent_bench.schemes import ClientSelectionScheme, UniformClientSelection
from decent_bench.utils.types import InitialStates

from ._fed_algorithm import FedAlgorithm

if TYPE_CHECKING:
    from decent_bench.agents import Agent
    from decent_bench.utils.array import Array


@dataclass(eq=False)
class FedOpt(FedAlgorithm, ABC):
    r"""
    Shared FedOpt template with client local SGD and server adaptive optimization :footcite:p:`Alg_FedOpt`.

    Each selected client starts from the broadcast global model :math:`\mathbf{x}_t` and performs
    ``num_local_epochs`` local SGD steps with client step size ``step_size``:

    .. math::
        \mathbf{x}_{i, t}^{(k+1)} = \mathbf{x}_{i, t}^{(k)} - \eta_l
        \nabla f_i(\mathbf{x}_{i, t}^{(k)}).

    After :math:`K` local steps, client :math:`i` forms the model delta

    .. math::
        \delta_i^t = \mathbf{x}_{i, t}^{(K)} - \mathbf{x}_t

    and uploads :math:`\delta_i^t` to the server. The server averages these client deltas uniformly:

    .. math::
        \Delta_t = \frac{1}{|S_t|} \sum_{i \in S_t} \delta_i^t.

    The server then applies the shared FedOpt first-moment and model updates

    .. math::
        \mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \Delta_t

    .. math::
        \mathbf{v}_t = \Phi(\mathbf{v}_{t-1}, \Delta_t)

    .. math::
        \mathbf{x}_{t+1} = \mathbf{x}_t + \eta
        \frac{\mathbf{m}_t}{\sqrt{\mathbf{v}_t} + \tau}.

    Here :math:`\eta_l` is the client learning rate (``step_size``), :math:`K` is the number of local SGD steps
    (``num_local_epochs``), :math:`\eta` is the server learning rate (``server_step_size``), :math:`\beta_1` is the
    first-moment coefficient, :math:`\tau` is the numerical stability term, and :math:`S_t` is the set of clients
    whose uploads are actually received in round :math:`t`. The second-moment update :math:`\Phi` is
    variant-specific and is defined by subclasses. Aggregation is always uniform across the received clients.
    Costs that preserve the :class:`~decent_bench.costs.EmpiricalRiskCost` abstraction use mini-batch local updates;
    generic costs use their usual full-gradient updates.

    .. footbibliography::
    """

    iterations: int = 100
    step_size: float = 0.001
    num_local_epochs: int = 1
    server_step_size: float = 0.001
    beta_1: float = 0.9
    tau: float = 1e-6
    selection_scheme: ClientSelectionScheme | None = field(
        default_factory=lambda: UniformClientSelection(client_fraction=1.0)
    )
    x0: InitialStates = None
    name: str = "FedOpt"

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
        if not (0 <= self.beta_1 < 1):
            raise ValueError("`beta_1` must satisfy 0 <= beta_1 < 1")
        if self.tau <= 0:
            raise ValueError("`tau` must be positive")

    def initialize(self, network: FedNetwork) -> None:
        self.x0 = initial_states(self.x0, network)
        server = network.server()
        server_x0 = self.x0[server]
        server.initialize(
            x=server_x0,
            aux_vars={"m": iop.zeros_like(server_x0), "v": iop.zeros_like(server_x0)},
        )
        for client in network.clients():
            client.initialize(x=self.x0[client])

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

    def _run_local_updates(self, network: FedNetwork, participating_clients: Sequence["Agent"]) -> None:
        server = network.server()
        for client in participating_clients:
            reference_x = self._get_server_broadcast(client, server)
            client.x = self._compute_local_update(client, server)
            network.send(sender=client, receiver=server, msg=client.x - reference_x)

    def _compute_local_update(self, client: "Agent", server: "Agent") -> "Array":
        """
        Run local SGD steps using the batching semantics of ``client.cost.gradient``.

        Costs that preserve the empirical-risk abstraction default ``gradient`` to ``indices="batch"``, so FedOpt
        variants perform mini-batch local updates automatically. Generic costs keep their usual full-gradient behavior.
        """
        local_x = self._get_server_broadcast(client, server)
        for _ in range(self.num_local_epochs):
            grad = client.cost.gradient(local_x)
            local_x -= self.step_size * grad
        return local_x

    def aggregate(
        self,
        network: FedNetwork,
        participating_clients: Sequence["Agent"],
    ) -> None:
        """
        Aggregate client model deltas uniformly, then apply the server adaptive optimizer.

        This method assumes clients upload final local model deltas. When used with
        :class:`~decent_bench.networks.Network` ``buffer_messages=True``, the caller must already have removed stale
        buffered client-to-server messages for the participating clients, so only current-round uploads are used.
        """
        server = network.server()
        received_clients = [client for client in participating_clients if client in server.messages]
        if not received_clients:
            return

        server_x = iop.copy(server.x)
        model_deltas = [server.messages[client] for client in received_clients]
        weights = [1.0] * len(received_clients)
        total_weight = float(len(received_clients))
        average_delta = self._weighted_average(model_deltas, weights, total_weight)

        server.aux_vars["m"] = self.beta_1 * server.aux_vars["m"] + (1 - self.beta_1) * average_delta
        server.aux_vars["v"] = self._update_second_moment(server.aux_vars["v"], average_delta)
        server.x = server_x + (
            self.server_step_size * server.aux_vars["m"] / (iop.sqrt(server.aux_vars["v"]) + self.tau)
        )

    @abstractmethod
    def _update_second_moment(self, second_moment: "Array", average_delta: "Array") -> "Array":
        """Return the updated server second-moment state for the current round."""
