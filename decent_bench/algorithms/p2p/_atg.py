from dataclasses import dataclass

import decent_bench.utils.interoperability as iop
from decent_bench.algorithms.utils import initial_states
from decent_bench.networks import P2PNetwork
from decent_bench.utils._tags import tags
from decent_bench.utils.types import InitialStates

from ._p2p_algorithm import P2PAlgorithm


@tags("peer-to-peer", "gradient-tracking", "dual method", "ADMM")
@dataclass(eq=False)
class ATG(P2PAlgorithm):
    r"""
    ADMM-Tracking Gradient (ATG) :footcite:p:`Alg_ATG` characterized by the update steps below.

    .. math::
        \begin{bmatrix} \mathbf{y}_{i,k} \\ \mathbf{s}_{i,k} \end{bmatrix} = \frac{1}{1 + \rho N_i}
        \left( \begin{bmatrix} \mathbf{x}_{i,k} \\ \nabla f_i(\mathbf{x}_{i,k}) \end{bmatrix}
        + \sum_j \mathbf{z}_{ij, k} \right)

    .. math::
        \mathbf{x}_{i,k+1} = (1 - \gamma) \mathbf{x}_{i,k}
        + \gamma \left( \mathbf{y}_{i,k} - \delta \mathbf{s}_{i,k} \right)

    .. math::
        \mathbf{z}_{ij, k+1} = (1-\alpha) \mathbf{z}_{ij, k} - \alpha \left( \mathbf{z}_{ji, k}
        - 2 \rho \begin{bmatrix} \mathbf{y}_{i,k} \\ \mathbf{s}_{i,k} \end{bmatrix} \right)

    where
    :math:`\mathbf{x}_{i, k} \in \mathbb{R}^n` is agent i's local optimization variable at iteration k,
    :math:`\mathbf{y}_{i,k}, \ \mathbf{s}_{i,k} \in \mathbb{R}^n`
    and :math:`\mathbf{z}_{ij,k} \in \mathbb{R}^{2n}` are auxiliary variables,
    :math:`N_i` is the number of neighbors of i,
    :math:`f_i` is i's local cost function,
    j is a neighbor of i. The parameters are: the penalty :math:`\rho > 0`, the relaxation :math:`\alpha \in (0, 1)`,
    the step-size :math:`\delta > 0`, and the mixing parameter :math:`\gamma > 0`. Notice that the convergence of
    the algorithm is guaranteed provided that :math:`\delta, \ \gamma` are below certain thresholds.

    The idea of the algorithm is to apply distributed ADMM to perform gradient tracking,
    instead of the usual average consensus.

    Aliases: :class:`ADMM_Tracking`, :class:`ADMM_TrackingGradient`

    .. footbibliography::

    """

    iterations: int = 100
    rho: float = 1
    alpha: float = 0.5
    gamma: float = 0.1
    delta: float = 0.001
    x0: InitialStates = None
    z0: InitialStates = None
    name: str = "ATG"

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
        if self.gamma <= 0:
            raise ValueError("`gamma` must be positive")
        if self.delta <= 0:
            raise ValueError("`delta` must be positive")

    def initialize(self, network: P2PNetwork) -> None:
        self.pN = {i: self.rho * len(network.neighbors(i)) for i in network.agents()}
        self.x0 = initial_states(self.x0, network)
        self.z0 = initial_states(self.z0, network)
        for i in network.agents():
            z_y0 = iop.stack([self.z0[i] for _ in network.neighbors(i)])
            z_s0 = iop.copy(z_y0)
            neighbor_to_idx = {j: idx for idx, j in enumerate(network.neighbors(i))}
            q = iop.zeros_like(self.x0[i])
            i.initialize(
                x=self.x0[i],
                aux_vars={"y": q, "s": q, "z_y": z_y0, "z_s": z_s0, "neighbor_to_idx": neighbor_to_idx},
            )

    def step(self, network: P2PNetwork, _: int) -> None:
        # step 1: update consensus-ADMM variables
        for i in network.active_agents():
            # update auxiliary variables
            i.aux_vars["y"] = (i.x + iop.sum(i.aux_vars["z_y"], dim=0)) / (1 + self.pN[i])
            i.aux_vars["s"] = (i.cost.gradient(i.x) + iop.sum(i.aux_vars["z_s"], dim=0)) / (1 + self.pN[i])
            # update local state
            i.x = (1 - self.gamma) * i.x + self.gamma * (i.aux_vars["y"] - self.delta * i.aux_vars["s"])

        # step 2: communicate and update z_{ij} variables
        for i in network.active_agents():
            for j in network.active_neighbors(i):
                # transmit the messages as a single message, stacking along the first axis
                idx = i.aux_vars["neighbor_to_idx"][j]
                s = iop.stack(
                    (
                        -i.aux_vars["z_y"][idx] + 2 * self.rho * i.aux_vars["y"],
                        -i.aux_vars["z_s"][idx] + 2 * self.rho * i.aux_vars["s"],
                    ),
                    dim=0,
                )
                network.send(i, j, s)

        for i in network.active_agents():
            for j, msg in i.messages.items():
                idx = i.aux_vars["neighbor_to_idx"][j]
                i.aux_vars["z_y"][idx] = (1 - self.alpha) * i.aux_vars["z_y"][idx] + self.alpha * msg[0]
                i.aux_vars["z_s"][idx] = (1 - self.alpha) * i.aux_vars["z_s"][idx] + self.alpha * msg[1]


ADMM_Tracking = ATG  # alias
ADMM_TrackingGradient = ATG  # alias
