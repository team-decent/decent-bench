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

    SolverArgs = dict[str, float]
else:
    SolverArgs = dict


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

    **Local solvers.**
    A selected client :math:`i` sets :math:`w^0_{i,k}=x_{i,k}` and
    :math:`v_{i,k}=2y_{k+1}-z_{i,k}`, then uses ``local_solver`` to approximately minimize the regularized local
    objective

    .. math::
        f_i(w) + \frac{1}{2\rho}\|w - v_{i,k}\|^2.

    The local gradient of this subproblem is

    .. math::
        \nabla f_i(w^\ell_{i,k}) + \frac{1}{\rho}(w^\ell_{i,k} - v_{i,k}).

    Costs preserving the :class:`~decent_bench.costs.EmpiricalRiskCost` abstraction use its default mini-batch
    sampling, so gradient-based local solvers use mini-batches. Generic :class:`~decent_bench.costs.Cost` objects use
    their normal full-gradient behavior. Solver-specific hyperparameters are passed through ``solver_args``.

    **Gradient descent.**
    The default ``local_solver="gd"`` uses ``step_size`` as the local step size and expects empty ``solver_args``:

    .. math::
        w^{\ell+1}_{i,k} = w^\ell_{i,k} - \gamma\left(\nabla f_i(w^\ell_{i,k})
        + \frac{1}{\rho}(w^\ell_{i,k} - v_{i,k})\right).

    **Nesterov.**
    The ``local_solver="nesterov"`` option applies a Nesterov-style update to the same local gradient. It initializes
    :math:`u_i^0=w^0_{i,k}` and uses ``step_size`` as the local step size. Its ``solver_args`` may contain
    ``"momentum"``; the default is ``0.9``:

    .. math::
        u_i^{\ell+1} = w^\ell_{i,k} - \gamma\left(\nabla f_i(w^\ell_{i,k})
        + \frac{1}{\rho}(w^\ell_{i,k} - v_{i,k})\right)

    .. math::
        w^{\ell+1}_{i,k} = u_i^{\ell+1}
        + \beta\left(u_i^{\ell+1} - u_i^\ell\right).

    One possible centralized strongly-convex choice is
    :math:`\beta=(\sqrt{L_i + 1/\rho} - \sqrt{\mu_i + 1/\rho}) / (\sqrt{L_i + 1/\rho}
    + \sqrt{\mu_i + 1/\rho})`, with local step size :math:`1/(L_i + 1/\rho)`, where
    :math:`L_i=\texttt{m_smooth}` and :math:`\mu_i=\texttt{m_cvx}`.

    **Adam.**
    The ``local_solver="adam"`` option applies Adam to the same local gradient. Adam moments are reset at the start
    of every local solve because Fed-LT locally trains the current round's subproblem rather than maintaining a
    persistent optimizer state across rounds. Its ``solver_args`` may contain ``"beta1"``, ``"beta2"``, and
    ``"epsilon"``; the defaults are ``0.9``, ``0.999``, and ``1e-8``:

    .. math::
        g^\ell_{i,k} =
        \nabla f_i(w^{\ell-1}_{i,k}) + \frac{1}{\rho}(w^{\ell-1}_{i,k} - v_{i,k})

    .. math::
        m^\ell_{i,k} = \beta_1 m^{\ell-1}_{i,k} + (1-\beta_1)g^\ell_{i,k}, \qquad
        s^\ell_{i,k} = \beta_2 s^{\ell-1}_{i,k} + (1-\beta_2)(g^\ell_{i,k})^2

    .. math::
        w^\ell_{i,k} =
        w^{\ell-1}_{i,k}
        - \gamma\frac{\hat m^\ell_{i,k}}{\sqrt{\hat s^\ell_{i,k}} + \epsilon},

    for :math:`\ell=1,\ldots,N_e`, with :math:`m^0_{i,k}=s^0_{i,k}=0`. The terms
    :math:`\hat m^\ell_{i,k}` and :math:`\hat s^\ell_{i,k}` are the Adam bias-corrected moments.


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
    local_solver: str = "gd"
    solver_args: SolverArgs = field(default_factory=dict)
    selection_scheme: ClientSelectionScheme | None = field(
        default_factory=lambda: UniformSelection(fraction_selected_clients=1.0)
    )
    x0: InitialStates = None
    z0: InitialStates = None
    name: str = "FedLT"

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
        if self.local_solver not in {"gd", "nesterov", "adam"}:
            raise ValueError("`local_solver` must be one of 'gd', 'nesterov', or 'adam'")
        self._validate_solver_args()

    def _validate_solver_args(self) -> None:
        user_args = self.solver_args
        if self.local_solver == "adam":
            default_args = {"beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8}
            self.solver_args = {
                "beta1": user_args.get("beta1", default_args["beta1"]),
                "beta2": user_args.get("beta2", default_args["beta2"]),
                "epsilon": user_args.get("epsilon", default_args["epsilon"]),
            }
            unknown_args = set(user_args) - set(default_args)
            if unknown_args:
                names = ", ".join(sorted(unknown_args))
                raise ValueError(f"Unsupported solver_args for local_solver='{self.local_solver}': {names}")
            if not (0 <= self.solver_args["beta1"] < 1):
                raise ValueError("`solver_args['beta1']` must satisfy 0 <= beta1 < 1")
            if not (0 <= self.solver_args["beta2"] < 1):
                raise ValueError("`solver_args['beta2']` must satisfy 0 <= beta2 < 1")
            if self.solver_args["epsilon"] <= 0:
                raise ValueError("`solver_args['epsilon']` must be positive")
        elif self.local_solver == "nesterov":
            default_args = {"momentum": 0.9}
            self.solver_args = {"momentum": user_args.get("momentum", default_args["momentum"])}
            unknown_args = set(user_args) - set(default_args)
            if unknown_args:
                names = ", ".join(sorted(unknown_args))
                raise ValueError(f"Unsupported solver_args for local_solver='{self.local_solver}': {names}")
            if not (0 <= self.solver_args["momentum"] < 1):
                raise ValueError("`solver_args['momentum']` must satisfy 0 <= momentum < 1")
        else:
            self.solver_args = {}
            unknown_args = set(user_args)
            if unknown_args:
                names = ", ".join(sorted(unknown_args))
                raise ValueError(f"Unsupported solver_args for local_solver='{self.local_solver}': {names}")

    def initialize(self, network: FedNetwork) -> None:
        self.x0 = initial_states(self.x0, network)
        self.z0 = self.x0 if self.z0 is None else initial_states(self.z0, network)

        server = network.server()
        z_by_client = {client: self.z0[client] for client in network.clients()}
        server.initialize(x=self.x0[server], aux_vars={"z_by_client": z_by_client})
        for client in network.clients():
            client.initialize(x=self.x0[client], aux_vars={"z": self.z0[client]})

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

        if self.local_solver == "nesterov":
            local_x = self._compute_nesterov_local_update(client, local_x, v)
        elif self.local_solver == "adam":
            local_x = self._compute_adam_local_update(client, local_x, v)
        else:
            for _ in range(self.num_local_epochs):
                grad = client.cost.gradient(local_x) + ((local_x - v) / self.rho)
                local_x -= self.step_size * grad

        z_next = z + (2 * (local_x - y))
        return local_x, z_next

    def _compute_nesterov_local_update(self, client: "Agent", local_x: "Array", v: "Array") -> "Array":
        momentum = self.solver_args["momentum"]
        u_previous = iop.copy(local_x)

        for _ in range(self.num_local_epochs):
            grad = client.cost.gradient(local_x) + ((local_x - v) / self.rho)
            u_next = local_x - (self.step_size * grad)
            local_x = u_next + (momentum * (u_next - u_previous))
            u_previous = u_next
        return local_x

    def _compute_adam_local_update(self, client: "Agent", local_x: "Array", v: "Array") -> "Array":
        beta1 = self.solver_args["beta1"]
        beta2 = self.solver_args["beta2"]
        epsilon = self.solver_args["epsilon"]
        m = iop.zeros_like(local_x)
        s = iop.zeros_like(local_x)

        for step in range(1, self.num_local_epochs + 1):
            grad = client.cost.gradient(local_x) + ((local_x - v) / self.rho)
            m = (beta1 * m) + ((1 - beta1) * grad)
            s = (beta2 * s) + ((1 - beta2) * (grad * grad))
            m_hat = m / (1 - (beta1**step))
            s_hat = s / (1 - (beta2**step))
            local_x -= self.step_size * m_hat / (iop.sqrt(s_hat) + epsilon)
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
