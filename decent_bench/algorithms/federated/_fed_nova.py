from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import decent_bench.utils.interoperability as iop
from decent_bench.algorithms.utils import infer_client_weight, initial_states
from decent_bench.networks import FedNetwork
from decent_bench.schemes import ClientSelectionScheme, UniformClientSelection
from decent_bench.utils._tags import tags
from decent_bench.utils.types import InitialStates, LocalSteps

from ._fed_algorithm import FedAlgorithm

if TYPE_CHECKING:
    from decent_bench.agents import Agent
    from decent_bench.utils.array import Array


@tags("federated")
@dataclass(eq=False)
class FedNova(FedAlgorithm):
    r"""
    FedNova with optional local momentum, proximal correction, and server momentum :footcite:p:`Alg_FedNova`.

    Each selected client starts from the broadcast global model :math:`\mathbf{x}_t = \mathbf{x}^{(t,0)}` and performs
    :math:`\tau_i` local steps with client step size ``step_size``. At local step :math:`k`, client :math:`i`
    computes the gradient

    .. math::
        \mathbf{g}_{i, t}^{(k)} = \nabla F_i(\mathbf{x}_{i, t}^{(k)}) + \mu
        \left(\mathbf{x}_{i, t}^{(k)} - \mathbf{x}_t\right),

    where the proximal term is present only when ``use_prox=True``. If local momentum is enabled,
    the momentum buffer and local direction update as

    .. math::
        \mathbf{v}_{i, t}^{(k+1)} = \beta \mathbf{v}_{i, t}^{(k)} + \mathbf{g}_{i, t}^{(k)},
        \qquad
        \mathbf{d}_{i, t}^{(k)} = \mathbf{v}_{i, t}^{(k+1)},

    otherwise :math:`\mathbf{d}_{i, t}^{(k)} = \mathbf{g}_{i, t}^{(k)}`. The local model update is

    .. math::
        \mathbf{x}_{i, t}^{(k+1)} = \mathbf{x}_{i, t}^{(k)} - \eta_l \mathbf{d}_{i, t}^{(k)}.

    Client :math:`i` accumulates the local update

    .. math::
        \mathbf{c}_i^t = \sum_{k=0}^{\tau_i - 1} \eta_l \mathbf{d}_{i, t}^{(k)},

    and maintains the FedNova scalar recurrences

    .. math::
        s_i^{(k+1)} = \beta s_i^{(k)} + 1

    when ``use_momentum=True``, and :math:`s_i^{(k+1)} = 1` otherwise, together with

    .. math::
        a_i^{(k+1)} = (1 - \eta_l \mu) a_i^{(k)} + s_i^{(k+1)}

    when ``use_prox=True`` and :math:`a_i^{(k+1)} = a_i^{(k)} + s_i^{(k+1)}` otherwise.
    During ``initialize``, the server resolves and stores each client's sample count :math:`n_i`.
    After :math:`\tau_i` local steps, client :math:`i` first uploads the FedNova coefficient :math:`a_i` and then
    uploads the cumulative local update :math:`\mathbf{c}_i^t` in a second transmission. For the clients in the
    current round whose two uploads are both actually received, the server forms the data-proportional client weight

    .. math::
        p_i = \frac{n_i}{\sum_{j \in S_t} n_j}.

    The server forms the weighted effective local-step coefficient

    .. math::
        \tau_{\mathrm{eff}, t} = \bar{a}_t = \sum_{i \in S_t} p_i a_i,

    and the normalized FedNova aggregate

    .. math::
        \mathbf{G}_t = \sum_{i \in S_t} p_i \frac{\tau_{\mathrm{eff}, t}}{a_i} \mathbf{c}_i^t.

    Without server momentum, the server update is

    .. math::
        \mathbf{x}_{t+1} = \mathbf{x}_t - \mathbf{G}_t.

    With ``use_server_momentum=True``, the server momentum buffer and model update become

    .. math::
        \mathbf{m}_{t+1} = \gamma \mathbf{m}_t + \mathbf{G}_t,
        \qquad
        \mathbf{x}_{t+1} = \mathbf{x}_t - \mathbf{m}_{t+1}.

    When ``use_momentum=False``, ``use_prox=False``, and ``use_server_momentum=False``, this reduces to the plain
    local-SGD FedNova variant. In that plain setting, FedNova reduces to FedAvg if and only if all participating
    clients use the same number of local steps (:math:`\tau_i = \tau_j` for all :math:`i, j \in S_t`) and
    FedNova and FedAvg both use data-proportional aggregation weights.

    Here :math:`\tau_i` is the number of local SGD steps used by client :math:`i`, which can be homogeneous
    (single int ``num_local_steps``) or heterogeneous via a mapping keyed by client agent. In this implementation,
    :math:`n_i` is inferred once during ``initialize`` from each client's local cost via
    :func:`~decent_bench.algorithms.utils.infer_client_weight`, then stored on the server for later rounds.
    If no first-phase ``a_i`` uploads are received in a round under network impairments, the server skips that round
    without error.

    .. footbibliography::
    """

    iterations: int = 100
    step_size: float = 0.001
    num_local_steps: LocalSteps = 1
    use_momentum: bool = False
    beta: float = 0.9
    use_prox: bool = False
    mu: float = 0.01
    use_server_momentum: bool = False
    gamma: float = 0.9
    selection_scheme: ClientSelectionScheme | None = field(
        default_factory=lambda: UniformClientSelection(client_fraction=1.0)
    )
    x0: InitialStates = None
    name: str = "FedNova"
    _num_local_steps_by_client: dict["Agent", int] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        """
        Validate hyperparameters.

        Raises:
            ValueError: if hyperparameters are invalid.

        """
        if self.step_size <= 0:
            raise ValueError("`step_size` must be positive")
        if not (0 <= self.beta < 1):
            raise ValueError("`beta` must satisfy 0 <= beta < 1")
        if self.mu < 0:
            raise ValueError("`mu` must be non-negative")
        if not (0 <= self.gamma < 1):
            raise ValueError("`gamma` must satisfy 0 <= gamma < 1")
        self._validate_num_local_steps()

    def _validate_num_local_steps(self) -> None:
        if isinstance(self.num_local_steps, int):
            if self.num_local_steps <= 0:
                raise ValueError("`num_local_steps` must be positive")
            return
        if isinstance(self.num_local_steps, dict):
            for step in self.num_local_steps.values():
                if step <= 0:
                    raise ValueError("`num_local_steps` must have positive values")
            return
        raise TypeError("`num_local_steps` must be an int or a mapping from Agent to integer values")

    def _settle_num_local_steps(self, network: FedNetwork) -> dict["Agent", int]:
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

    def _resolve_client_sample_counts(self, network: FedNetwork) -> dict["Agent", float]:
        client_sample_counts: dict[Agent, float] = {}
        for client in network.clients():
            client_sample_counts[client] = infer_client_weight(client)
        return client_sample_counts

    def initialize(self, network: FedNetwork) -> None:
        self.x0 = initial_states(self.x0, network)
        server = network.server()
        server_x0 = self.x0[server]
        aux_vars: dict[str, Any] = {
            "client_sample_counts": self._resolve_client_sample_counts(network),
            "received_a_i": {},
        }
        if self.use_server_momentum:
            aux_vars["m"] = iop.zeros_like(server_x0)
        server.initialize(x=server_x0, aux_vars=aux_vars)
        for client in network.clients():
            client.initialize(x=self.x0[client])
        self._num_local_steps_by_client = self._settle_num_local_steps(network)
        self.num_local_steps = self._num_local_steps_by_client

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
        self._collect_received_normalizers(network, participating_clients)
        if not network.server().aux_vars["received_a_i"]:
            for client in participating_clients:
                client.aux_vars.pop("_fednova_cumulative_gradient", None)
            return
        self._clear_buffered_server_messages(network, participating_clients)
        self._communicate_cumulative_gradients(network, participating_clients)
        self.aggregate(network, participating_clients)

    def _run_local_updates(self, network: FedNetwork, participating_clients: Sequence["Agent"]) -> None:
        server = network.server()
        for client in participating_clients:
            local_x, cumulative_gradient, a_i = self._compute_local_update(client, server)
            client.x = local_x
            client.aux_vars["_fednova_cumulative_gradient"] = cumulative_gradient
            normalizer_upload = iop.reshape(iop.to_array_like(a_i, cumulative_gradient), (1,))
            network.send(sender=client, receiver=server, msg=normalizer_upload)

    def _compute_local_update(self, client: "Agent", server: "Agent") -> tuple["Array", "Array", float]:
        """
        Run local SGD steps and return the cumulative local SGD update and FedNova coefficient ``a_i``.

        Costs that preserve the empirical-risk abstraction default ``gradient`` to ``indices="batch"``, so FedNova
        performs mini-batch local updates automatically. Generic costs keep their usual full-gradient behavior. This
        method assumes ``initialize`` has already normalized ``num_local_steps`` to a per-client mapping.

        """
        reference_x = self._get_server_broadcast(client, server)
        local_x = iop.copy(reference_x)
        cumulative_gradient = iop.zeros_like(reference_x)
        local_momentum = iop.zeros_like(reference_x)
        tau_i = self._num_local_steps_by_client[client]
        a_i = 0.0
        momentum_scalar = 0.0

        for _ in range(tau_i):
            grad = client.cost.gradient(local_x)
            if self.use_prox:
                grad += self.mu * (local_x - reference_x)

            if self.use_momentum:
                local_momentum = (self.beta * local_momentum) + grad
                direction = local_momentum
            else:
                direction = grad

            local_step_update = self.step_size * direction
            local_x -= local_step_update
            cumulative_gradient += local_step_update

            momentum_scalar = (self.beta * momentum_scalar) + 1.0 if self.use_momentum else 1.0

            if self.use_prox:
                a_i = ((1 - (self.step_size * self.mu)) * a_i) + momentum_scalar
            else:
                a_i += momentum_scalar

        return local_x, cumulative_gradient, a_i

    def _collect_received_normalizers(self, network: FedNetwork, participating_clients: Sequence["Agent"]) -> None:
        server = network.server()
        received_normalizers = {
            client: iop.astype(server.messages[client], float)
            for client in participating_clients
            if client in server.messages
        }
        server.aux_vars["received_a_i"] = received_normalizers

    def _communicate_cumulative_gradients(self, network: FedNetwork, participating_clients: Sequence["Agent"]) -> None:
        server = network.server()
        for client in participating_clients:
            cumulative_gradient = client.aux_vars.pop("_fednova_cumulative_gradient")
            network.send(sender=client, receiver=server, msg=cumulative_gradient)

    def aggregate(
        self,
        network: FedNetwork,
        participating_clients: Sequence["Agent"],
    ) -> None:
        r"""
        Aggregate FedNova client uploads following the Local-SGD FedNova pseudocode.

        This method assumes the current round has already cached the received FedNova coefficients ``a_i`` on the
        server and that the server's current messages contain the received cumulative local updates
        :math:`\mathbf{c}_i^t`. Client sample counts are looked up from the mapping stored on the server during
        ``initialize``. When used with :class:`~decent_bench.networks.Network` ``buffer_messages=True``, the caller
        must already have removed stale buffered client-to-server messages for the participating clients at the start
        of the round and between the ``a_i`` and cumulative-gradient upload phases. If no client has both uploads
        available in the current round, this method returns without updating the server model.

        Raises:
            ValueError: if any received FedNova coefficient ``a_i`` is non-positive.

        """
        server = network.server()
        received_normalizers = server.aux_vars["received_a_i"]
        received_gradient_clients = [client for client in participating_clients if client in server.messages]
        received_clients = [client for client in received_gradient_clients if client in received_normalizers]
        if not received_clients:
            return

        server_sample_counts = server.aux_vars["client_sample_counts"]
        server_x = iop.copy(server.x)
        cumulative_gradients = [server.messages[client] for client in received_clients]
        a_values = [received_normalizers[client] for client in received_clients]
        if any(a_i <= 0 for a_i in a_values):
            raise ValueError("FedNova coefficients `a_i` must be positive")

        sample_counts = [server_sample_counts[client] for client in received_clients]
        total_samples = float(sum(sample_counts))
        client_weights = [n_i / total_samples for n_i in sample_counts]

        tau_eff = sum(client_weight * a_i for client_weight, a_i in zip(client_weights, a_values, strict=True))
        weighted_terms = [
            client_weight * (tau_eff / a_i) * cumulative_gradient
            for cumulative_gradient, a_i, client_weight in zip(
                cumulative_gradients, a_values, client_weights, strict=True
            )
        ]
        global_update = iop.sum(iop.stack(weighted_terms, dim=0), dim=0)
        if self.use_server_momentum:
            server.aux_vars["m"] = (self.gamma * server.aux_vars["m"]) + global_update
            server.x = server_x - server.aux_vars["m"]
        else:
            server.x = server_x - global_update
