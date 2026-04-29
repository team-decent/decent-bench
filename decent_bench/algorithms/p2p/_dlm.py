from dataclasses import dataclass

import decent_bench.utils.interoperability as iop
from decent_bench.agents import Agent
from decent_bench.algorithms.utils import initial_states
from decent_bench.networks import P2PNetwork
from decent_bench.utils._tags import tags
from decent_bench.utils.array import Array
from decent_bench.utils.types import InitialStates

from ._p2p_algorithm import P2PAlgorithm


@tags("peer-to-peer", "ADMM", "gradient-based")
@dataclass(eq=False)
class DLM(P2PAlgorithm):
    r"""
    Decentralized Linearized ADMM (DLM) :footcite:p:`Alg_DLM_1, Alg_DLM_2` characterized by the update steps below.

    .. math::
        \mathbf{x}_{i,k+1} = \mathbf{x}_{i,k} - \mu \left( \nabla f_i(\mathbf{x}_{i,k})
        + \rho \sum_j (\mathbf{x}_{i,k} - \mathbf{x}_{j,k}) + \mathbf{z}_{i,k} \right)

    .. math::
        \mathbf{z}_{i, k+1} = \mathbf{z}_{i, k} + \rho \sum_j (\mathbf{x}_{i,k+1} - \mathbf{x}_{j,k+1})

    where
    :math:`\mathbf{x}_{i, k}` is agent i's local optimization variable at iteration k,
    :math:`\mathbf{z}_{i,k}` is the local dual variable,
    :math:`f_i` is i's local cost function,
    j is a neighbor of i. The parameters are: the penalty :math:`\rho > 0` and the step-size :math:`\mu > 0`.

    Alias: :class:`DecentralizedLinearizedADMM`

    .. footbibliography::

    """

    iterations: int = 100
    step_size: float = 0.001
    penalty: float = 1
    x0: InitialStates = None
    name: str = "DLM"

    def __post_init__(self) -> None:
        """
        Validate hyperparameters.

        Raises:
            ValueError: if hyperparameters are invalid.

        """
        if self.step_size <= 0:
            raise ValueError("`step_size` must be positive")
        if self.penalty <= 0:
            raise ValueError("`penalty` must be positive")

    def initialize(self, network: P2PNetwork) -> None:
        self.x0 = initial_states(self.x0, network)
        for i in network.agents():
            y = iop.zeros_like(self.x0[i])  # y must be initialized to zero
            i.initialize(x=self.x0[i], aux_vars={"y": y, "s": y})

    def step(self, network: P2PNetwork, iteration: int) -> None:
        if iteration == 0:
            # step 0: first communication round
            for i in network.active_agents():
                network.broadcast(i, i.x)

            # compute and store \sum_j (\mathbf{x}_{i,0} - \mathbf{x}_{j,0})
            for i in network.active_agents():
                s = self._sum_messages(i)
                if s is not None:
                    i.aux_vars["s"] = len(i.messages) * i.x - s
        else:
            # step 1: update primal variable
            for i in network.active_agents():
                i.x = i.x - self.step_size * (  # noqa: PLR6104
                    i.cost.gradient(i.x) + self.penalty * i.aux_vars["s"] + i.aux_vars["y"]
                )

            # step 2: communication round
            for i in network.active_agents():
                network.broadcast(i, i.x)

            # compute and store \sum_j (\mathbf{x}_{i,k+1} - \mathbf{x}_{j,k+1})
            for i in network.active_agents():
                s = self._sum_messages(i)
                if s is not None:
                    i.aux_vars["s"] = len(i.messages) * i.x - s

            # step 3: update dual variable
            for i in network.active_agents():
                i.aux_vars["y"] += self.penalty * i.aux_vars["s"]

    def _sum_messages(self, i: "Agent") -> "Array | None":
        s = None
        for val in i.messages.values():
            if s is None:
                s = val
            else:
                s += val
        return s


DecentralizedLinearizedADMM = DLM  # alias
