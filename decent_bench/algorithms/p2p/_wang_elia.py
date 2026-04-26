from dataclasses import dataclass

import decent_bench.utils.interoperability as iop
from decent_bench.algorithms.utils import initial_states
from decent_bench.networks import P2PNetwork
from decent_bench.utils._tags import tags
from decent_bench.utils.types import InitialStates

from ._p2p_algorithm import P2PAlgorithm


@tags("peer-to-peer", "gradient-tracking")
@dataclass(eq=False)
class WangElia(P2PAlgorithm):
    r"""
    Wang-Elia gradient tracking algorithm characterized by the updates below, see :footcite:p:`Alg_Wang_1, Alg_Wang_2`.

    .. math::
        \mathbf{x}_{i, k+1} = \mathbf{x}_{i, k} - \sum_j \mathbf{K}_{ij} (\mathbf{x}_{j, k} + \mathbf{z}_{j, k})
                            - \rho \nabla f_i(\mathbf{x}_{i,k})
    .. math::
        \mathbf{z}_{i, k+1} = \mathbf{z}_{i, k} + \sum_j \mathbf{K}_{ij} \mathbf{x}_{j, k}

    where
    :math:`\mathbf{x}_{i, k}` is agent i's local optimization variable at iteration k,
    :math:`\rho` is the step size,
    :math:`f_i` is agent i's local cost function,
    j is a neighbor of i or i itself,
    and :math:`\mathbf{K}_{ij}` is the weight between agent i and j.
    The matrix :math:`\mathbf{K}` is chosen as :math:`0.5 (\mathbf{I} - \mathbf{W})`,
    where :math:`\mathbf{W}` is the Metropolis weight matrix.

    .. footbibliography::

    """

    iterations: int = 100
    step_size: float = 0.001
    x0: InitialStates = None
    name: str = "Wang-Elia"

    def __post_init__(self) -> None:
        """
        Validate hyperparameters.

        Raises:
            ValueError: if hyperparameters are invalid.

        """
        if self.step_size <= 0:
            raise ValueError("`step_size` must be positive")

    def initialize(self, network: P2PNetwork) -> None:
        self.x0 = initial_states(self.x0, network)
        for i in network.agents():
            i.initialize(x=self.x0[i], aux_vars={"z": iop.zeros_like(self.x0[i]), "x_old": self.x0[i]})

        W = network.weights  # noqa: N806
        K = 0.5 * (iop.eye_like(W) - W)  # noqa: N806

        self.K = K

    def step(self, network: P2PNetwork, _: int) -> None:
        # 1st communication round
        for i in network.active_agents():
            msg = i.x + i.aux_vars["z"]
            i.aux_vars["msg"] = msg
            network.broadcast(i, msg)

        # do consensus and local gradient step
        for i in network.active_agents():
            neighborhood_avg = self.K[i, i] * i.aux_vars["msg"]
            for j, msg_j in i.messages.items():
                neighborhood_avg += self.K[i, j] * msg_j

            i.aux_vars["x_old"] = i.x
            i.x = i.x - neighborhood_avg - self.step_size * i.cost.gradient(i.x)

        # 2nd communication round
        for i in network.active_agents():
            network.broadcast(i, i.aux_vars["x_old"])

        # update auxiliary variable
        for i in network.active_agents():
            neighborhood_avg = self.K[i, i] * i.aux_vars["x_old"]
            for j, x_old_j in i.messages.items():
                neighborhood_avg += self.K[i, j] * x_old_j
            i.aux_vars["z"] += neighborhood_avg
