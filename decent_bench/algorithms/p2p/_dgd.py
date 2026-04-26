from dataclasses import dataclass

from decent_bench.algorithms.utils import initial_states
from decent_bench.networks import P2PNetwork
from decent_bench.utils._tags import tags
from decent_bench.utils.types import InitialStates

from ._p2p_algorithm import P2PAlgorithm


@tags("peer-to-peer", "gradient-based")
@dataclass(eq=False)
class DGD(P2PAlgorithm):
    r"""
    Distributed gradient descent characterized by the update step below.

    .. math::
        \mathbf{x}_{i, k+1} = \gamma (\sum_{j} \mathbf{W}_{ij} \mathbf{x}_{j,k}) - \rho \nabla f_i(\mathbf{x}_{i,k})

    where
    :math:`\mathbf{x}_{i, k}` is agent i's local optimization variable at iteration k,
    j is a neighbor of i or i itself,
    :math:`\mathbf{W}_{ij}` is the metropolis weight between agent i and j,
    :math:`\rho` is the step size,
    and :math:`f_i` is agent i's local cost function.

    """

    iterations: int = 100
    step_size: float = 0.001
    x0: InitialStates = None
    name: str = "DGD"

    def __post_init__(self) -> None:
        """
        Validate hyperparameters.

        Raises:
            ValueError: if hyperparameters are invalid.

        """
        if self.step_size <= 0:
            raise ValueError("`step_size` must be positive")

    def initialize(self, network: P2PNetwork) -> None:
        x0 = initial_states(self.x0, network)
        for i in network.agents():
            i.initialize(x=x0[i])

        self.W = network.weights

    def step(self, network: P2PNetwork, _: int) -> None:
        for i in network.active_agents():
            network.broadcast(i, i.x)

        for i in network.active_agents():
            neighborhood_avg = self.W[i, i] * i.x
            for j, x_j in i.messages.items():
                neighborhood_avg += self.W[i, j] * x_j
            i.x = neighborhood_avg - self.step_size * i.cost.gradient(i.x)
