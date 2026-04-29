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
class FedProx(FedAlgorithm):
    r"""
    Federated Proximal (FedProx) with local SGD epochs :footcite:p:`Alg_FedProx`.

    Each client solves a proximalized local subproblem around the round's server model:

    .. math::
        h_k(\mathbf{w}; \mathbf{w}^t) = F_k(\mathbf{w}) + \frac{\mu}{2} \|\mathbf{w} - \mathbf{w}^t\|^2

    .. math::
        \mathbf{x}_{i, k}^{(t+1)} = \mathbf{x}_{i, k}^{(t)} - \eta
        \nabla h_k(\mathbf{x}_{i, k}^{(t)}; \mathbf{w}^t)

    where :math:`\nabla h_k(\mathbf{w}; \mathbf{w}^t) = \nabla F_k(\mathbf{w}) + \mu (\mathbf{w} - \mathbf{w}^t)`.

    .. math::
        \mathbf{x}_{k+1} = \frac{1}{|S_k|} \sum_{i \in S_k} \mathbf{x}_{i, k}^{(E)}

    where :math:`\mathbf{w}^t` is the server model broadcast at the start of round :math:`k`, held fixed
    throughout each selected client's local epochs, :math:`\mu \geq 0` is the proximal coefficient,
    :math:`\eta` is the step size, and :math:`S_k` is the set of participating clients. Setting ``mu=0.0``
    recovers :class:`FedAvg <decent_bench.algorithms.federated.FedAvg>` exactly. Aggregation uses uniform averaging
    over the participating clients. Client selection defaults to uniform sampling with fraction 1.0. For
    :class:`~decent_bench.costs.EmpiricalRiskCost`, local updates use mini-batches of size
    :attr:`EmpiricalRiskCost.batch_size <decent_bench.costs.EmpiricalRiskCost.batch_size>`; for generic costs,
    local updates use full-batch gradients.

    .. footbibliography::
    """

    iterations: int = 100
    step_size: float = 0.001
    num_local_epochs: int = 1
    mu: float = 0.01
    selection_scheme: ClientSelectionScheme | None = field(
        default_factory=lambda: UniformClientSelection(client_fraction=1.0)
    )
    x0: InitialStates = None
    name: str = "FedProx"

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
        if self.mu < 0:
            raise ValueError("`mu` must be non-negative")

    def initialize(self, network: FedNetwork) -> None:
        self.x0 = initial_states(self.x0, network)
        network.server().initialize(x=self.x0[network.server()])
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
        for client in participating_clients:
            client.x = self._compute_local_update(client, network.server())
            network.send(sender=client, receiver=network.server(), msg=client.x)

    def _compute_local_update(self, client: "Agent", server: "Agent") -> "Array":
        """
        Run local proximal gradient steps using the batching semantics of ``client.cost.gradient``.

        Costs that preserve the empirical-risk abstraction default ``gradient`` to ``indices="batch"``, so FedProx
        performs mini-batch local updates automatically. Generic costs keep their usual full-gradient behavior.
        """
        reference_x = self._get_server_broadcast(client, server)
        local_x = iop.copy(reference_x)
        for _ in range(self.num_local_epochs):
            grad = client.cost.gradient(local_x) + self.mu * (local_x - reference_x)
            local_x -= self.step_size * grad
        return local_x
