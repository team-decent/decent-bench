from dataclasses import dataclass

import decent_bench.utils.interoperability as iop
from decent_bench.algorithms.utils import initial_states
from decent_bench.networks import P2PNetwork
from decent_bench.utils._tags import tags
from decent_bench.utils.types import InitialStates

from ._p2p_algorithm import P2PAlgorithm


@tags("peer-to-peer", "gradient-tracking")
@dataclass(eq=False)
class AugDGM(P2PAlgorithm):
    r"""
    Aug-DGM :footcite:p:`Alg_Aug_DMG` or ATC-DIGing :footcite:p:`Alg_ATC_DIG` gradient tracking algorithm.

    The algorithm is characterized by the updates below.

    .. math::
        \mathbf{x}_{i, k+1} = \sum_j \mathbf{W}_{ij} (\mathbf{x}_{j, k} - \rho \mathbf{y}_{j, k})
    .. math::
        \mathbf{y}_{i, k+1} = \sum_j \mathbf{W}_{ij} (\mathbf{y}_{j, k}
                            + \nabla f_j(\mathbf{x}_{j,k+1}) - \nabla f_j(\mathbf{x}_{j,k}))

    where
    :math:`\mathbf{x}_{i, k}` is agent i's local optimization variable at iteration k,
    :math:`\rho` is the step size,
    :math:`f_i` is agent i's local cost function,
    j is a neighbor of i or i itself,
    and :math:`\mathbf{W}_{ij}` is the metropolis weight between agent i and j.

    Alias: :class:`ATC_DIGing`

    .. footbibliography::

    """

    iterations: int = 100
    step_size: float = 0.001
    x0: InitialStates = None
    name: str = "Aug-DGM"

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
            y0 = i.cost.gradient(self.x0[i])
            z = iop.zeros_like(self.x0[i])
            i.initialize(x=self.x0[i], aux_vars={"y": y0, "g": y0, "g_new": z, "s": z})

        self.W = network.weights

    def step(self, network: P2PNetwork, _: int) -> None:
        # 1st communication round
        #     step 1: perform local gradient step and communicate
        for i in network.active_agents():
            i.aux_vars["s"] = i.x - self.step_size * i.aux_vars["y"]

        for i in network.active_agents():
            network.broadcast(i, i.aux_vars["s"])

        #     step 2: update state and compute new local gradient
        for i in network.active_agents():
            neighborhood_avg = self.W[i, i] * i.aux_vars["s"]
            for j, s_j in i.messages.items():
                neighborhood_avg += self.W[i, j] * s_j
            i.x = neighborhood_avg
            i.aux_vars["g_new"] = i.cost.gradient(i.x)

        # 2nd communication round
        #     step 1: transmit local gradient tracker
        for i in network.active_agents():
            msg = i.aux_vars["y"] + i.aux_vars["g_new"] - i.aux_vars["g"]
            i.aux_vars["msg"] = msg
            network.broadcast(i, msg)

        #     step 2: update y (global gradient estimator)
        for i in network.active_agents():
            neighborhood_avg = self.W[i, i] * i.aux_vars["msg"]
            for j, q_j in i.messages.items():
                neighborhood_avg += self.W[i, j] * q_j
            i.aux_vars["y"] = neighborhood_avg
            i.aux_vars["g"] = i.aux_vars["g_new"]


ATC_DIGing = AugDGM  # alias
