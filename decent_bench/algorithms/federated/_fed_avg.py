from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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
class FedAvg(FedAlgorithm):
    r"""
    Federated Averaging (FedAvg) with local SGD epochs :footcite:p:`Alg_FedAvg`.

    .. math::
        \mathbf{x}_{i, k}^{(t+1)} = \mathbf{x}_{i, k}^{(t)} - \eta \nabla f_i(\mathbf{x}_{i, k}^{(t)})

    .. math::
        \mathbf{x}_{k+1} = \frac{1}{|S_k|} \sum_{i \in S_k} \mathbf{x}_{i, k}^{(E)}

    where :math:`t` indexes the local training epochs on each client and :math:`E` is the number of local epochs per
    round (``num_local_epochs``), :math:`\eta` is the step size, and :math:`S_k` is the set of participating clients at
    round :math:`k`. In FedAvg, each selected client performs ``num_local_epochs`` local SGD epochs, then the server
    aggregates the final local models to form :math:`\mathbf{x}_{k+1}` using uniform averaging over the participating
    clients. Client selection (subsampling) defaults to uniform sampling with fraction 1.0 (all active clients) and can
    be customized via ``selection_scheme``. Costs that preserve the
    :class:`~decent_bench.costs.EmpiricalRiskCost` abstraction use client-side mini-batches of size
    :attr:`EmpiricalRiskCost.batch_size <decent_bench.costs.EmpiricalRiskCost.batch_size>`; generic cost wrappers fall
    back to full-gradient local updates.

    .. footbibliography::
    """

    # C=0.1; batch size= inf/10/50 (dataset sizes are bigger; normally 1/10 of the total dataset).
    # E= 5/20 (num local epochs).
    iterations: int = 100
    step_size: float = 0.001
    num_local_epochs: int = 1
    selection_scheme: ClientSelectionScheme | None = field(
        default_factory=lambda: UniformClientSelection(client_fraction=1.0)
    )
    x0: InitialStates = None
    name: str = "FedAvg"

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
        Run local gradient steps using the batching semantics of ``client.cost.gradient``.

        Costs that preserve the empirical-risk abstraction default ``gradient`` to ``indices="batch"``, so FedAvg
        performs mini-batch local updates automatically. Generic costs keep their usual full-gradient behavior.
        """
        local_x = self._get_server_broadcast(client, server)
        for _ in range(self.num_local_epochs):
            grad = client.cost.gradient(local_x)
            local_x -= self.step_size * grad
        return local_x
