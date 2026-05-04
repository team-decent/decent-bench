import math
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
class FedLT(FedAlgorithm):
    r"""
    Federated Local Training (Fed-LT) with cost-driven local gradients :footcite:p:`Alg_FedPLT,Alg_FedLT`.

    Fed-LT maintains one auxiliary variable :math:`z_i` per client. At the start of round :math:`k`, the server
    computes the broadcast variable

    .. math::
        y_{k+1} = \operatorname{prox}_{\rho h / N}\left(\frac{1}{N}\sum_{i=1}^N z_{i,k}\right)

    where :math:`N` is the number of clients. The server sends :math:`y_{k+1}` to the selected clients.
    The initial auxiliary states ``z0`` follow the same :obj:`~decent_bench.utils.types.InitialStates` convention as
    ``x0``. If ``z0`` is ``None``, Fed-LT initializes :math:`z_{i,0}=x_{i,0}` for every client.

    In this implementation, each client cost :math:`f_i` is treated as the full local objective already available on
    that client. The global regularizer :math:`h` is represented
    by the server cost's proximal operator; with the default :class:`~decent_bench.costs.ZeroCost` server, this is the
    documented :math:`h=0` case and the server step is plain averaging.

    A selected client :math:`i` sets :math:`w^0_{i,k}=x_{i,k}` and :math:`v_{i,k}=2y_{k+1}-z_{i,k}`, then performs
    ``num_local_epochs`` local steps. The default local update uses

    .. math::
        w^{\ell+1}_{i,k} = w^\ell_{i,k} - \gamma\left(\nabla f_i(w^\ell_{i,k})
        + \frac{1}{\rho}(w^\ell_{i,k} - v_{i,k})\right).

    The extra term :math:`(w^\ell_{i,k} - v_{i,k}) / \rho` is the quadratic local-training penalty from the Fed-LT
    update. Costs preserving the :class:`~decent_bench.costs.EmpiricalRiskCost` abstraction use its default
    mini-batch sampling, so this behaves as local SGD. Generic :class:`~decent_bench.costs.Cost` objects use their
    normal full-gradient behavior, so this behaves as local GD.

    If ``use_acceleration`` is true, Fed-LT uses the accelerated local update with each client cost's ``m_smooth`` and
    ``m_cvx`` metadata. It initializes :math:`u_i^0=w^0_{i,k}`. With :math:`L_i=\texttt{m_smooth}` and
    :math:`\mu_i=\texttt{m_cvx}`, it applies

    .. math::
        u_i^{\ell+1} = w^\ell_{i,k} - \frac{1}{L_i + 1/\rho}\left(\nabla f_i(w^\ell_{i,k})
        + \frac{1}{\rho}(w^\ell_{i,k} - v_{i,k})\right)

    .. math::
        w^{\ell+1}_{i,k} = u_i^{\ell+1}
        + \frac{\sqrt{L_i + 1/\rho} - \sqrt{\mu_i + 1/\rho}}
        {\sqrt{L_i + 1/\rho} + \sqrt{\mu_i + 1/\rho}}
        \left(u_i^{\ell+1} - u_i^\ell\right).

    If the client cost has :math:`\mu_i=0`, this coefficient reduces to the Fed-LT expression with
    :math:`\sqrt{1/\rho}` in the second term. The constants must be finite and non-negative. The gradient call remains
    cost-driven: empirical costs use their default mini-batches and generic costs use full gradients.

    After local training, the client sets

    .. math::
        x_{i,k+1}=w^{N_e}_{i,k}, \qquad z_{i,k+1}=z_{i,k}+2(x_{i,k+1}-y_{k+1}),

    and uploads :math:`z_{i,k+1}` to the server. Inactive clients keep
    :math:`x_{i,k+1}=x_{i,k}` and :math:`z_{i,k+1}=z_{i,k}`. For later server averages, the server stores a received
    fresh :math:`z_{i,k+1}` when the upload arrives and otherwise keeps its previous stored :math:`z_i`.

    Fed-PLT is the privacy-noise version of Fed-LT :footcite:p:`Alg_FedPLT`.

    .. footbibliography::
    """

    iterations: int = 100
    step_size: float = 0.001
    num_local_epochs: int = 1
    rho: float = 1.0
    use_acceleration: bool = False
    selection_scheme: ClientSelectionScheme | None = field(
        default_factory=lambda: UniformClientSelection(client_fraction=1.0)
    )
    x0: InitialStates = None
    z0: InitialStates = None
    name: str = "FedLT"
    _accelerated_smoothness: dict["Agent", float] = field(init=False, default_factory=dict, repr=False)
    _accelerated_momentum: dict["Agent", float] = field(init=False, default_factory=dict, repr=False)

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
        if self.rho <= 0:
            raise ValueError("`rho` must be positive")

    def initialize(self, network: FedNetwork) -> None:
        if self.use_acceleration:
            self._initialize_accelerated_parameters(network)

        self.x0 = initial_states(self.x0, network)
        self.z0 = self.x0 if self.z0 is None else initial_states(self.z0, network)

        server = network.server()
        z_by_client = {client: self.z0[client] for client in network.clients()}
        server.initialize(x=self.x0[server], aux_vars={"z_by_client": z_by_client})
        for client in network.clients():
            client.initialize(x=self.x0[client], aux_vars={"z": self.z0[client]})

    def _initialize_accelerated_parameters(self, network: FedNetwork) -> None:
        self._accelerated_smoothness = {}
        self._accelerated_momentum = {}
        for client in network.clients():
            m_smooth = client.cost.m_smooth
            m_cvx = client.cost.m_cvx
            if not math.isfinite(m_smooth) or not math.isfinite(m_cvx) or m_smooth < 0 or m_cvx < 0:
                raise ValueError(
                    "`use_acceleration=True` requires finite non-negative `m_smooth` and `m_cvx` on every client cost"
                )
            if m_smooth < m_cvx:
                raise ValueError("`use_acceleration=True` requires `m_smooth >= m_cvx` on every client cost")

            smoothness = m_smooth + (1 / self.rho)
            strong_convexity = m_cvx + (1 / self.rho)
            sqrt_smoothness = math.sqrt(smoothness)
            sqrt_strong_convexity = math.sqrt(strong_convexity)
            self._accelerated_smoothness[client] = smoothness
            self._accelerated_momentum[client] = (sqrt_smoothness - sqrt_strong_convexity) / (
                sqrt_smoothness + sqrt_strong_convexity
            )

    def step(self, network: FedNetwork, iteration: int) -> None:
        y = self._compute_server_y(network)
        network.server().x = y

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

    def _compute_server_y(self, network: FedNetwork) -> "Array":
        z_values = list(network.server().aux_vars["z_by_client"].values())
        average_z = iop.mean(iop.stack(z_values, dim=0), dim=0)
        return network.server().cost.proximal(average_z, self.rho / len(network.clients()))

    def _run_local_updates(self, network: FedNetwork, participating_clients: Sequence["Agent"]) -> None:
        for client in participating_clients:
            client.x, client.aux_vars["z"] = self._compute_local_update(client, network.server())
            network.send(sender=client, receiver=network.server(), msg=client.aux_vars["z"])

    def _compute_local_update(self, client: "Agent", server: "Agent") -> tuple["Array", "Array"]:
        """
        Run Fed-LT local training and return the updated local model and auxiliary variable.

        The gradient call intentionally delegates batching to ``client.cost.gradient``. For
        :class:`~decent_bench.costs.EmpiricalRiskCost`, that default call samples mini-batches; for generic costs it
        is a full-gradient call.
        """
        y = self._get_server_broadcast(client, server)
        z = client.aux_vars["z"]
        v = (2 * y) - z
        local_x = iop.copy(client.x)

        if self.use_acceleration:
            local_x = self._compute_accelerated_local_update(client, local_x, v)
        else:
            for _ in range(self.num_local_epochs):
                grad = client.cost.gradient(local_x) + ((local_x - v) / self.rho)
                local_x -= self.step_size * grad

        z_next = z + (2 * (local_x - y))
        return local_x, z_next

    def _compute_accelerated_local_update(self, client: "Agent", local_x: "Array", v: "Array") -> "Array":
        smoothness = self._accelerated_smoothness[client]
        momentum = self._accelerated_momentum[client]
        u_previous = iop.copy(local_x)

        for _ in range(self.num_local_epochs):
            grad = client.cost.gradient(local_x) + ((local_x - v) / self.rho)
            u_next = local_x - ((1 / smoothness) * grad)
            local_x = u_next + (momentum * (u_next - u_previous))
            u_previous = u_next
        return local_x

    def aggregate(
        self,
        network: FedNetwork,
        participating_clients: Sequence["Agent"],
    ) -> None:
        """
        Store received Fed-LT ``z`` uploads for future server averages.

        Clients whose uploads are not received keep their previous server-side ``z`` value, matching the stale-value
        aggregation in partial participation and lossy communication settings.
        """
        z_by_client = network.server().aux_vars["z_by_client"]
        for client in participating_clients:
            if client in network.server().messages:
                z_by_client[client] = network.server().messages[client]
