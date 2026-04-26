from dataclasses import dataclass

import decent_bench.utils.interoperability as iop
from decent_bench.algorithms.utils import initial_states
from decent_bench.networks import P2PNetwork
from decent_bench.utils._tags import tags
from decent_bench.utils.types import InitialStates

from ._p2p_algorithm import P2PAlgorithm


@tags("peer-to-peer", "dual method", "ADMM")
@dataclass(eq=False)
class ADMM(P2PAlgorithm):
    r"""
    Distributed Alternating Direction Method of Multipliers characterized by the update step below.

    .. math::
        \mathbf{x}_{i, k+1} = \operatorname{prox}_{\frac{1}{\rho N_i} f_i}
        \left(\sum_j \mathbf{z}_{ij, k} \frac{1}{\rho N_i} \right)
    .. math::
        \mathbf{z}_{ij, k+1} = (1-\alpha) \mathbf{z}_{ij, k} - \alpha (\mathbf{z}_{ji, k} - 2 \rho \mathbf{x}_{j, k+1})

    where
    :math:`\mathbf{x}_{i, k}` is agent i's local optimization variable at iteration k,
    :math:`\operatorname{prox}` is the proximal operator described in :meth:`Cost.proximal()
    <decent_bench.costs.Cost.proximal>`,
    :math:`\rho > 0` is the Lagrangian penalty parameter,
    :math:`N_i` is the number of neighbors of i,
    :math:`f_i` is i's local cost function,
    j is a neighbor of i,
    and :math:`\alpha \in (0, 1)` is the relaxation parameter.

    Note:
        ``x0`` and ``z0`` follow the :obj:`~decent_bench.utils.types.InitialStates` convention and are resolved
        per agent during ``initialize`` via
        :func:`~decent_bench.algorithms.utils.initial_states`.
        If ``x0`` is ``None`` and ``z0`` is provided, each agent initializes ``x0`` from ``z0`` with one proximal
        update:

        .. math::
            x_{i,0} = \operatorname{prox}_{\frac{1}{\rho N_i} f_i}\left(\frac{z_{i,0}}{\rho}\right)

        The :math:`\mathbf{z}_{ij}` variables of an agent are all initialized to
        the same value specified in ``z0`` (if any).

    """

    iterations: int = 100
    rho: float = 1
    alpha: float = 0.5
    x0: InitialStates = None
    z0: InitialStates = None
    name: str = "ADMM"

    def __post_init__(self) -> None:
        """
        Validate hyperparameters.

        Raises:
            ValueError: if hyperparameters are invalid.

        """
        if self.rho <= 0:
            raise ValueError("`rho` must be positive")
        if not (0 < self.alpha < 1):
            raise ValueError("`alpha` must be in (0, 1)")

    def initialize(self, network: P2PNetwork) -> None:
        self.rho_i = {i: 1 / (self.rho * len(network.neighbors(i))) for i in network.agents()}
        x_from_z = self.x0 is None and self.z0 is not None  # if x0 needs to be created from z0
        self.x0 = initial_states(self.x0, network)
        self.z0 = initial_states(self.z0, network)
        for i in network.agents():
            if x_from_z:
                self.x0[i] = i.cost.proximal(x=self.z0[i] / self.rho, rho=self.rho_i[i])
            z0 = iop.stack([self.z0[i] for _ in network.neighbors(i)])
            neighbor_to_idx = {j: idx for idx, j in enumerate(network.neighbors(i))}
            i.initialize(x=self.x0[i], aux_vars={"z": z0, "neighbor_to_idx": neighbor_to_idx})

    def step(self, network: P2PNetwork, _: int) -> None:
        for i in network.active_agents():
            i.x = i.cost.proximal(x=iop.sum(i.aux_vars["z"], dim=0) * self.rho_i[i], rho=self.rho_i[i])

        for i in network.active_agents():
            for j in network.active_neighbors(i):
                idx = i.aux_vars["neighbor_to_idx"][j]
                network.send(i, j, i.aux_vars["z"][idx] - 2 * self.rho * i.x)

        for i in network.active_agents():
            for j, msg in i.messages.items():
                idx = i.aux_vars["neighbor_to_idx"][j]
                i.aux_vars["z"][idx] = (1 - self.alpha) * i.aux_vars["z"][idx] - self.alpha * (msg)
