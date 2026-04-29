from dataclasses import dataclass
from typing import TYPE_CHECKING

import decent_bench.utils.interoperability as iop
from decent_bench.agents import Agent
from decent_bench.algorithms.utils import initial_states
from decent_bench.costs import EmpiricalRiskCost
from decent_bench.networks import P2PNetwork
from decent_bench.utils._tags import tags

from ._lt_admm import LT_ADMM


@tags("peer-to-peer", "gradient-based", "dual method", "ADMM", "variance-reduction")
@dataclass(eq=False)
class LT_ADMM_VR(LT_ADMM):  # noqa: N801
    """
    Local Training ADMM with Variance Reduction (LT-ADMM-VR) :footcite:p:`Alg_LT_ADMM_VR`.

    Extends LT-ADMM with variance reduction techniques for improved convergence.
    This variant implements additional gradient variance reduction mechanisms
    during the local training phase.

    Args:
        iterations: Total number of communication rounds (K)
        num_local_steps: Number of local training steps (tau)
        step_size: Local step size (gamma)
        aux_step_size: Local step size (beta)
        penalty: Penalty parameter (rho)
        alpha: Relaxation parameter (alpha)
        x0: Initial parameters (optional)
        v2: Whether to use the LT-ADMM-VR-2 variant with improved
            variance reduction techniques which is less computational heavy (default True).
        name: Algorithm name (default "LT-ADMM-VR")

    Raises:
        TypeError: If any agent's cost function is not an instance of EmpiricalRiskCost.

    .. footbibliography::

    """

    v2: bool = True  # Whether to use the LT-ADMM-VR-2 variant
    name: str = "LT-ADMM-VR"

    def initialize(self, network: P2PNetwork) -> None:
        self.x0 = initial_states(self.x0, network)

        # Initialize agents with auxiliary variables
        for i in network.agents():
            if not isinstance(i.cost, EmpiricalRiskCost):
                raise TypeError("LT-ADMM-VR is only compatible with EmpiricalRiskCost.")

            neighbors = network.neighbors(i)
            z_i = iop.zeros(
                shape=(len(neighbors), *iop.shape(self.x0[i])),
                framework=i.cost.framework,
                device=i.cost.device,
            )
            neighbor_to_idx: dict[Agent, int] = {}  # Mapping from neighbor to index in z_i array

            for idx, j in enumerate(neighbors):
                z_i[idx] = iop.copy(self.x0[i])
                neighbor_to_idx[j] = idx

            r_grads = i.cost.gradient(self.x0[i], indices="all", reduction=None) if self.v2 else None
            # Initialize auxiliary variables for LT-ADMM
            aux_vars = {
                "phi": self.x0[i],  # phi_i,k - model parameters
                "r_grads": r_grads,  # shape (m_i, dim) - nabla f_{i,h}(r_{i,h,k})
                "z_i": z_i,  # z_ij,k+1 - auxiliary consensus variable
                "neighbor_to_idx": neighbor_to_idx,
            }
            i.initialize(x=self.x0[i], aux_vars=aux_vars)

    def _local_training(self, agent: Agent, network: P2PNetwork) -> None:
        """
        Enhanced local training with variance reduction.

        Raises:
            TypeError: If the agent's cost is not an instance of EmpiricalRiskCost, as LT-ADMM-VR is only compatible
            with EmpiricalRiskCost.

        """
        if TYPE_CHECKING:
            if not isinstance(agent.cost, EmpiricalRiskCost):
                raise TypeError("LT-ADMM-VR is only compatible with EmpiricalRiskCost.")

        agent.aux_vars["phi"] = iop.copy(agent.x)
        z_sum = iop.sum(agent.aux_vars["z_i"], dim=0)
        # Always use the number of neighbors for the penalty term to ensure proper scaling
        multiplier = self.penalty * len(network.neighbors(agent))
        correction = self.aux_step_size * (multiplier * agent.x - z_sum)

        if not self.v2:
            r_grads = agent.cost.gradient(agent.x, indices="all", reduction=None)
            agent.aux_vars["r_grads"] = r_grads

        for _ in range(self.num_local_steps):
            batch_grad = agent.cost.gradient(agent.aux_vars["phi"])
            batch_used = agent.cost.batch_used
            r_grads = iop.mean(agent.aux_vars["r_grads"][batch_used], dim=0)
            current_gradient = (batch_grad - r_grads) + iop.mean(agent.aux_vars["r_grads"], dim=0)

            step = self.step_size * current_gradient + correction
            agent.aux_vars["phi"] -= step

            r_grads = agent.cost.gradient(agent.aux_vars["phi"], indices=batch_used, reduction=None)
            agent.aux_vars["r_grads"][batch_used] = r_grads

        # Update agent's main parameter (line 10)
        agent.x = agent.aux_vars["phi"]
