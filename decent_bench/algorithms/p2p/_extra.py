from dataclasses import dataclass

import decent_bench.utils.interoperability as iop
from decent_bench.algorithms.utils import initial_states
from decent_bench.networks import P2PNetwork
from decent_bench.utils._tags import tags
from decent_bench.utils.types import InitialStates

from ._p2p_algorithm import P2PAlgorithm


@tags("peer-to-peer", "gradient-tracking")
@dataclass(eq=False)
class EXTRA(P2PAlgorithm):
    r"""
    EXTRA :footcite:p:`Alg_EXTRA` gradient tracking algorithm characterized by the update steps below.

    .. math::
        \mathbf{x}_{i, k+1}
        = \mathbf{x}_{i, k} + \sum_j \mathbf{W}_{ij} \mathbf{x}_{j,k}
        - \sum_j \tilde{\mathbf{W}}_{ij} \mathbf{x}_{j,k-1}
        - \rho (\nabla f_i(\mathbf{x}_{i,k}) - \nabla f_i(\mathbf{x}_{i,k-1}))

    where
    :math:`\mathbf{x}_{i, k}` is agent i's local optimization variable at iteration k,
    :math:`\rho` is the step size,
    :math:`f_i` is agent i's local cost function,
    j is a neighbor of i or i itself,
    :math:`\mathbf{W}_{ij}` is the metropolis weight between agent i and j,
    and :math:`\tilde{\mathbf{W}} = (\mathbf{I} + \mathbf{W}) / 2`.

    .. footbibliography::

    """

    iterations: int = 100
    step_size: float = 0.001
    x0: InitialStates = None
    name: str = "EXTRA"

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
            z = iop.zeros_like(self.x0[i])
            i.initialize(
                x=self.x0[i],
                aux_vars={"x_old": self.x0[i], "x_old_old": z, "x_cons": z},
            )

        self.W = network.weights

    def step(self, network: P2PNetwork, iteration: int) -> None:
        if iteration == 0:
            # first iteration (iteration k=1)
            for i in network.active_agents():
                network.broadcast(i, i.x)

            for i in network.active_agents():
                neighborhood_avg = self.W[i, i] * i.x
                for j, x_j in i.messages.items():
                    neighborhood_avg += self.W[i, j] * x_j
                i.aux_vars["x_cons"] = neighborhood_avg  # store W x_k
                i.aux_vars["x_old"] = i.x  # store x_0
                i.x = neighborhood_avg - self.step_size * i.cost.gradient(i.x)
        else:
            # subsequent iterations (k >= 2)
            for i in network.active_agents():
                network.broadcast(i, i.x)

            for i in network.active_agents():
                neighborhood_avg = self.W[i, i] * i.x
                for j, x_j in i.messages.items():
                    neighborhood_avg += self.W[i, j] * x_j
                i.aux_vars["x_old_old"] = i.aux_vars["x_old"]  # store x_{k-1}
                i.aux_vars["x_old"] = i.x  # store x_k
                # update x_{k+1}
                i.x = (
                    i.x
                    + neighborhood_avg
                    - 0.5 * i.aux_vars["x_old_old"]
                    - 0.5 * i.aux_vars["x_cons"]
                    - self.step_size * (i.cost.gradient(i.x) - i.cost.gradient(i.aux_vars["x_old_old"]))
                )
                i.aux_vars["x_cons"] = neighborhood_avg  # store W x_k
