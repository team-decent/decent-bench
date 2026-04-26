from dataclasses import dataclass

from decent_bench.algorithms.utils import initial_states
from decent_bench.networks import P2PNetwork
from decent_bench.utils._tags import tags
from decent_bench.utils.types import InitialStates

from ._p2p_algorithm import P2PAlgorithm


@tags("peer-to-peer", "gradient-tracking")
@dataclass(eq=False)
class SimpleGT(P2PAlgorithm):
    r"""
    Gradient tracking algorithm characterized by the update step below.

    .. math::
        \mathbf{y}_{i, k+1} = \mathbf{x}_{i, k} - \rho \nabla f_i(\mathbf{x}_{i,k})
    .. math::
        \mathbf{x}_{i, k+1} = \mathbf{y}_{i, k+1} - \mathbf{y}_{i, k} + \sum_j \mathbf{W}_{ij} \mathbf{x}_{j,k}

    where
    :math:`\mathbf{x}_{i, k}` is agent i's local optimization variable at iteration k,
    :math:`\rho` is the step size,
    :math:`f_i` is agent i's local cost function,
    j is a neighbor of i or i itself,
    and :math:`\mathbf{W}_{ij}` is the metropolis weight between agent i and j.

    Alias: :class:`SimpleGradientTracking`

    """

    iterations: int = 100
    step_size: float = 0.001
    x0: InitialStates = None
    name: str = "SimpleGT"

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
            i.initialize(x=self.x0[i], aux_vars={"y": self.x0[i]})

        self.W = network.weights

    def step(self, network: P2PNetwork, _: int) -> None:
        for i in network.active_agents():
            network.broadcast(i, i.x)

        for i in network.active_agents():
            i.aux_vars["y_new"] = i.x - self.step_size * i.cost.gradient(i.x)

            neighborhood_avg = self.W[i, i] * i.x
            for j, x_j in i.messages.items():
                neighborhood_avg += self.W[i, j] * x_j

            i.x = i.aux_vars["y_new"] - i.aux_vars["y"] + neighborhood_avg
            i.aux_vars["y"] = i.aux_vars["y_new"]


SimpleGradientTracking = SimpleGT  # Alias
