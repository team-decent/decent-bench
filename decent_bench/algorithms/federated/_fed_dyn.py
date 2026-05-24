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


@tags("federated")
@dataclass(eq=False)
class FedDyn(FedAlgorithm):
    r"""
    Federated Dynamic Regularization (FedDyn) with local gradient steps :footcite:p:`Alg_FedDyn`.

    FedDyn keeps one dynamic state :math:`g_i` per client and one server auxiliary vector :math:`h`. In each
    communication round, the paper writes the selected-client update as the exact minimizer of the
    dynamic-regularized local objective

    .. math::
        f_i(\theta) - \langle g_i, \theta \rangle
        + \frac{\alpha}{2}\|\theta - \theta^t\|^2

    In practice, and following the local SGD device update used in the paper's experiments, this implementation
    approximates that minimization by running ``num_local_steps`` gradient steps
    from the received server model, with local gradient

    .. math::
        \nabla f_i(\theta) - g_i + \alpha(\theta - \theta^t).

    After local training, each participating client updates its dynamic state as

    .. math::
        g_i^+ = g_i - \alpha(\theta_i^+ - \theta^t).

    The server aggregates only the selected client models it actually receives. If :math:`R_t` is the received subset,
    :math:`m` is the total number of clients, and :math:`\theta^t` is the server model before aggregation, then

    .. math::
        h^+ = h - \frac{\alpha}{m}\sum_{i \in R_t}(\theta_i^+ - \theta^t),
        \qquad
        \theta^+ = \frac{1}{|R_t|}\sum_{i \in R_t}\theta_i^+ - \frac{1}{\alpha}h^+.

    Here :math:`\alpha` is the dynamic regularization coefficient (the corresponding argument is ``penalty``), and the
    local step size is the scalar used in local SGD (the corresponding argument is ``step_size``).

    If no selected client model is received, the server model and ``h`` remain unchanged. Unselected clients and
    selected clients that miss the server broadcast keep their previous local model and dynamic state. Costs that
    preserve the :class:`~decent_bench.costs.EmpiricalRiskCost` abstraction use mini-batch local updates; generic costs
    keep their usual full-gradient behavior. Client selection defaults to uniform sampling with fraction 1.0.

    .. footbibliography::
    """

    iterations: int = 100
    step_size: float = 0.001
    penalty: float = 0.01
    num_local_steps: int = 1
    selection_scheme: ClientSelectionScheme | None = field(
        default_factory=lambda: UniformSelection(fraction_selected_clients=1.0)
    )
    x0: InitialStates = None
    name: str = "FedDyn"

    def __post_init__(self) -> None:
        """
        Validate hyperparameters.

        Raises:
            ValueError: if hyperparameters are invalid.

        """
        if self.step_size <= 0:
            raise ValueError("`step_size` must be positive")
        if self.penalty <= 0:
            raise ValueError("`penalty` must be positive")
        if self.num_local_steps <= 0:
            raise ValueError("`num_local_steps` must be positive")

    def initialize(self, network: FedNetwork) -> None:
        self.x0 = initial_states(self.x0, network)
        server = network.server()
        server_x0 = self.x0[server]
        server.initialize(x=server_x0, aux_vars={"h": iop.zeros_like(server_x0)})
        for client in network.clients():
            client_x0 = self.x0[client]
            client.initialize(x=client_x0, aux_vars={"g": iop.zeros_like(client_x0)})

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

    def _run_local_updates(self, network: FedNetwork, participating_clients: Sequence["Agent"]) -> None:
        for client in participating_clients:
            reference_x = self._get_server_broadcast(client, network.server())
            local_x = self._compute_local_update(client, reference_x)
            client.x = local_x
            client.aux_vars["g"] -= self.penalty * (local_x - reference_x)
            network.send(sender=client, receiver=network.server(), msg=client.x)

    def _compute_local_update(self, client: "Agent", reference_x: "Array") -> "Array":
        """
        Run local FedDyn gradient steps using the batching semantics of ``client.cost.gradient``.

        Costs that preserve the empirical-risk abstraction default ``gradient`` to ``indices="batch"``, so FedDyn
        performs mini-batch local updates automatically. Generic costs keep their usual full-gradient behavior.
        """
        local_x = iop.copy(reference_x)
        dynamic_state = iop.copy(client.aux_vars["g"])
        for _ in range(self.num_local_steps):
            grad = client.cost.gradient(local_x) - dynamic_state + self.penalty * (local_x - reference_x)
            local_x -= self.step_size * grad
        return local_x

    def aggregate(
        self,
        network: FedNetwork,
        participating_clients: Sequence["Agent"],
    ) -> None:
        """
        Aggregate received FedDyn client models and apply the server dynamic correction.

        Only client models received in the current round are aggregated.
        """
        server = network.server()
        received_clients = [client for client in participating_clients if client in server.messages()]
        if not received_clients:
            return
        reference_x = iop.copy(server.x)
        client_models = [server.message(client) for client in received_clients]
        average_model = self._weighted_average(
            client_models,
            weights=[1.0] * len(received_clients),
            total_weight=float(len(received_clients)),
        )
        model_delta_sum = iop.zeros_like(reference_x)
        for model in client_models:
            model_delta_sum += model - reference_x
        server.aux_vars["h"] -= self.penalty * model_delta_sum / len(network.clients())
        server.x = average_model - server.aux_vars["h"] / self.penalty
