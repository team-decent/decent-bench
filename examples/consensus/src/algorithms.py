from dataclasses import dataclass
from decent_bench.networks import P2PNetwork
from decent_bench.algorithms.p2p import P2PAlgorithm
import decent_bench.utils.interoperability as iop
from decent_bench.algorithms.utils import initial_states


Y_UPDATE = "y"
Z_UPDATE = "z"


@dataclass(eq=False)
class AvgConsensus(P2PAlgorithm):
    r"""
    Average consensus characterized by the update step below.

    .. math::
        \mathbf{x}_{i, k+1} = \sum_{j} \mathbf{W}_{ij} \mathbf{x}_{j,k}, \qquad \mathbf{x}_{i, 0} = \mathbf{u}_i

    where
    :math:`\mathbf{x}_{i, k}` is agent i's local optimization variable at iteration k,
    j is a neighbor of i or i itself,
    :math:`\mathbf{W}_{ij}` is the metropolis weight between agent i and j.

    """

    iterations: int = 100
    name: str = "Average Consensus"

    def initialize(self, network: P2PNetwork) -> None:  # noqa: D102
        self.x0 = initial_states({a: a.data["u"] for a in network.agents()}, network)
        for i in network.agents():
            i.initialize(x=self.x0[i])

        self.W = network.weights

    def step(self, network: P2PNetwork, _: int) -> None:  # noqa: D102
        for i in network.active_agents():
            network.broadcast(i, i.x)

        for i in network.active_agents():
            neighborhood_avg = self.W[i, i] * i.x
            if len(i.messages()) > 0:
                for j, x_j in i.messages().items():
                    neighborhood_avg += self.W[i, j] * x_j
            i.x = neighborhood_avg


@dataclass(eq=False)
class RatioConsensus(P2PAlgorithm):
    r"""
    Ratio consensus characterized by the update step below.

    .. math::
        \mathbf{x}_{i, k+1} = \sum_{j} \mathbf{W}_{ij} \mathbf{x}_{j,k}, \qquad \mathbf{x}_{i, 0} = \mathbf{u}_i

    where
    :math:`\mathbf{x}_{i, k}` is agent i's local optimization variable at iteration k,
    j is a neighbor of i or i itself,
    :math:`\mathbf{W}_{ij}` is the metropolis weight between agent i and j.

    """

    iterations: int = 100
    name: str = "Ratio Consensus"

    def initialize(self, network: P2PNetwork) -> None:  # noqa: D102
        self.x0 = initial_states({a: a.data["u"] for a in network.agents()}, network)
        for i in network.agents():
            i.initialize(x=self.x0[i], aux_vars={"y": self.x0[i], "z": iop.ones_like(self.x0[i])})

        self.W = network.weights

    def step(self, network: P2PNetwork, _: int) -> None:  # noqa: D102
        for i in network.active_agents():
            network.broadcast(i, i.aux_vars["y"], channel=Y_UPDATE)
            network.broadcast(i, i.aux_vars["z"], channel=Z_UPDATE)

        for i in network.active_agents():
            y_avg = self.W[i, i] * i.aux_vars["y"]
            if len(i.messages(channel=Y_UPDATE)) > 0:
                for j, y_j in i.messages(channel=Y_UPDATE).items():
                    y_avg += self.W[i, j] * y_j

            z_avg = self.W[i, i] * i.aux_vars["z"]
            if len(i.messages(channel=Z_UPDATE)) > 0:
                for j, z_j in i.messages(channel=Z_UPDATE).items():
                    z_avg += self.W[i, j] * z_j

            i.x = iop.div(y_avg, z_avg)
