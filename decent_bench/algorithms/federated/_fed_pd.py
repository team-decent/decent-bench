import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import decent_bench.utils.interoperability as iop
from decent_bench.algorithms.utils import initial_states
from decent_bench.networks import FedNetwork
from decent_bench.utils._tags import tags
from decent_bench.utils.types import InitialStates, LocalSteps

from ._fed_algorithm import FedAlgorithm

if TYPE_CHECKING:
    from decent_bench.agents import Agent
    from decent_bench.utils.array import Array


@tags("federated")
@dataclass(eq=False)
class FedPD(FedAlgorithm):
    r"""
    Federated Primal-Dual (FedPD) with local gradient steps :footcite:p:`Alg_FedPD`.

    FedPD rewrites federated learning as a global consensus problem with client primal variables
    :math:`\mathbf{x}_i`, local centre variables :math:`\mathbf{x}_{0,i}`, and dual variables
    :math:`\lambda_i`. In each round, all active clients run ``num_local_steps`` gradient steps on the local
    augmented Lagrangian

    .. math::
        f_i(\mathbf{x}_i) + \langle \lambda_i, \mathbf{x}_i - \mathbf{x}_{0,i} \rangle
        + \frac{1}{2 \eta}\|\mathbf{x}_i - \mathbf{x}_{0,i}\|^2.

    The local gradient is

    .. math::
        \nabla f_i(\mathbf{x}_i) + \lambda_i
        + \frac{1}{\eta}(\mathbf{x}_i - \mathbf{x}_{0,i}).

    After the local gradient steps, clients update their dual variables and local centre candidates:

    .. math::
        \lambda_i^+ = \lambda_i + \frac{1}{\eta}(\mathbf{x}_i^+ - \mathbf{x}_{0,i}),
        \qquad
        \mathbf{x}_{0,i}^+ = \mathbf{x}_i^+ + \eta \lambda_i^+.

    With probability ``1 - skip_probability``, clients upload their local centre candidates and the server uniformly
    averages the candidates it actually receives. If at least one candidate is received, the server centre is then
    sent back to all active clients; clients that do not receive the server's synchronized centre keep their local
    centre candidate. With probability ``skip_probability``, aggregation is skipped and every participating client
    keeps its local centre candidate.

    Costs that preserve the :class:`~decent_bench.costs.EmpiricalRiskCost` abstraction use mini-batch local updates;
    generic costs keep their usual full-gradient behavior. ``num_local_steps`` can be homogeneous (single integer) or
    heterogeneous via a mapping keyed by client agent. Partial client participation is not supported.

    .. footbibliography::
    """

    iterations: int = 100
    step_size: float = 0.001
    eta: float = 1.0
    skip_probability: float = 0.0
    num_local_steps: LocalSteps = 1
    x0: InitialStates = None
    name: str = "FedPD"
    _num_local_steps_by_client: dict["Agent", int] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        """
        Validate hyperparameters.

        Raises:
            ValueError: if hyperparameters are invalid.

        """
        if self.step_size <= 0:
            raise ValueError("`step_size` must be positive")
        if self.eta <= 0:
            raise ValueError("`eta` must be positive")
        if not (0 <= self.skip_probability <= 1):
            raise ValueError("`skip_probability` must satisfy 0 <= skip_probability <= 1")
        self._validate_num_local_steps()

    def initialize(self, network: FedNetwork) -> None:
        self.x0 = initial_states(self.x0, network)
        server = network.server()
        server_x0 = self.x0[server]
        server.initialize(x=server_x0)
        for client in network.clients():
            client_x0 = self.x0[client]
            client.initialize(
                x=client_x0,
                aux_vars={
                    "lambda": iop.zeros_like(client_x0),
                    "center": iop.copy(server_x0),
                },
            )
        self._num_local_steps_by_client = self._settle_num_local_steps(network)
        self.num_local_steps = self._num_local_steps_by_client

    def step(self, network: FedNetwork, iteration: int) -> None:
        participating_clients = self.select_clients(network, iteration)
        if not participating_clients:
            return

        self._run_local_updates(participating_clients)
        if random.random() >= self.skip_probability:
            self._communicate_center_candidates(network, participating_clients)
            self.aggregate(network, participating_clients)

    def _run_local_updates(self, participating_clients: Sequence["Agent"]) -> None:
        for client in participating_clients:
            previous_center = iop.copy(client.aux_vars["center"])
            local_x = self._compute_local_update(client)
            new_dual = client.aux_vars["lambda"] + (local_x - previous_center) / self.eta
            client.x = local_x
            client.aux_vars["lambda"] = new_dual
            client.aux_vars["center"] = local_x + self.eta * new_dual

    def _compute_local_update(self, client: "Agent") -> "Array":
        """
        Run local FedPD gradient steps using the batching semantics of ``client.cost.gradient``.

        Costs that preserve the empirical-risk abstraction default ``gradient`` to ``indices="batch"``, so FedPD
        performs mini-batch local updates automatically. Generic costs keep their usual full-gradient behavior. This
        method assumes ``initialize`` has already normalized ``num_local_steps`` to a per-client mapping.
        """
        local_x = iop.copy(client.x)
        center = iop.copy(client.aux_vars["center"])
        dual = iop.copy(client.aux_vars["lambda"])
        for _ in range(self._num_local_steps_by_client[client]):
            grad = client.cost.gradient(local_x) + dual + (local_x - center) / self.eta
            local_x -= self.step_size * grad
        return local_x

    def _communicate_center_candidates(
        self,
        network: FedNetwork,
        participating_clients: Sequence["Agent"],
    ) -> None:
        for client in participating_clients:
            network.send(sender=client, receiver=network.server(), msg=client.aux_vars["center"])

    def aggregate(
        self,
        network: FedNetwork,
        participating_clients: Sequence["Agent"],
    ) -> None:
        """
        Aggregate received FedPD centre candidates and broadcast the synchronized centre.

        Unlike most federated algorithms, a FedPD communication round couples aggregation with centre
        synchronization: after the server averages the received centre candidates, it immediately broadcasts the
        updated centre back to all participating clients.

        If no centre candidates are received, the server state is left unchanged and no synchronization is sent.
        """
        server = network.server()
        received_clients = [client for client in participating_clients if client in network.server().messages]
        if not received_clients:
            return
        center_candidates = [server.messages[client] for client in received_clients]
        weights = [1.0] * len(received_clients)
        total_weight = float(len(received_clients))
        server.x = self._weighted_average(center_candidates, weights, total_weight)
        self.server_broadcast(network, participating_clients)
        for client in participating_clients:
            if server in client.messages:
                client.aux_vars["center"] = self._get_server_broadcast(client, server)
