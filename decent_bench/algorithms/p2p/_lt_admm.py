from dataclasses import dataclass

import decent_bench.utils.interoperability as iop
from decent_bench.agents import Agent
from decent_bench.algorithms.utils import initial_states
from decent_bench.networks import P2PNetwork
from decent_bench.utils._tags import tags
from decent_bench.utils.types import InitialStates

from ._p2p_algorithm import P2PAlgorithm


@tags("peer-to-peer", "gradient-based", "dual method", "ADMM")
@dataclass(eq=False)
class LT_ADMM(P2PAlgorithm):  # noqa: N801
    """
    Local Training ADMM (LT-ADMM) :footcite:p:`Alg_LT_ADMM_VR`.

    Args:
        iterations: Total number of communication rounds (K)
        num_local_steps: Number of local training steps (tau)
        step_size: Local step size (gamma)
        aux_step_size: Local step size (beta)
        penalty: Penalty parameter (rho)
        alpha: Relaxation parameter (alpha)
        x0: Initial parameters (optional)
        name: Algorithm name (default "LT-ADMM")

    .. footbibliography::

    """

    iterations: int = 100  # Total number of communication rounds (K)
    num_local_steps: int = 5  # Number of local training steps (tau)
    step_size: float = 0.01  # Local step size (gamma)
    aux_step_size: float = 0.01  # Local step size (beta)
    penalty: float = 1.0  # Penalty parameter (rho)
    alpha: float = 0.5  # Relaxation parameter (alpha)
    x0: InitialStates = None  # Initial parameters (optional)
    name: str = "LT-ADMM"

    def __post_init__(self) -> None:
        """
        Validate parameters.

        Raises:
            ValueError: If any of the parameters are invalid (e.g., non-positive iterations, local_steps,
            step_size, penalty, or alpha).

        """
        if self.num_local_steps <= 0:
            raise ValueError("local_steps must be positive")
        if isinstance(self.step_size, float) and self.step_size <= 0:
            raise ValueError("step_size must be positive")
        if isinstance(self.aux_step_size, float) and self.aux_step_size <= 0:
            raise ValueError("aux_step_size must be positive")
        if self.penalty <= 0:
            raise ValueError("penalty must be positive")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")

    def initialize(self, network: P2PNetwork) -> None:
        x0 = initial_states(self.x0, network)

        # Initialize agents with auxiliary variables
        for i in network.agents():
            neighbors = network.neighbors(i)
            z_i = iop.zeros(
                shape=(len(neighbors), *iop.shape(x0[i])),
                framework=i.cost.framework,
                device=i.cost.device,
            )
            neighbor_to_idx: dict[Agent, int] = {}  # Mapping from neighbor to index in z_i array

            for idx, j in enumerate(neighbors):
                z_i[idx] = iop.copy(x0[i])
                neighbor_to_idx[j] = idx

            aux_vars = {
                "phi": x0[i],  # phi_i,k - model parameters
                "z_i": z_i,  # z_ij,k+1 - auxiliary consensus variable
                "neighbor_to_idx": neighbor_to_idx,
            }
            i.initialize(x=x0[i], aux_vars=aux_vars)

    def step(self, network: P2PNetwork, _: int) -> None:
        # Step 1: Local training phase
        for i in network.active_agents():
            self._local_training(i, network)

        # Step 2: Communication phase
        for i in network.active_agents():
            self._communication(i, network)

        # Step 3: Auxiliary update phase
        for i in network.active_agents():
            self._auxiliary_update(i)

    def _local_training(self, agent: Agent, network: P2PNetwork) -> None:
        """
        Perform local training steps (Algorithm 1, lines 2-9).

        Updates phi_i,k and gradient estimators r_i,h,k.
        """
        agent.aux_vars["phi"] = iop.copy(agent.x)
        z_sum = iop.sum(agent.aux_vars["z_i"], dim=0)
        # Always use the number of neighbors for the penalty term to ensure proper scaling
        multiplier = self.penalty * len(network.neighbors(agent))
        correction = self.aux_step_size * (multiplier * agent.x - z_sum)

        for _ in range(self.num_local_steps):
            current_gradient = agent.cost.gradient(agent.aux_vars["phi"])
            step = self.step_size * current_gradient + correction
            # Update phi_i,k according to gradient step (line 7)
            agent.aux_vars["phi"] -= step

        # Update agent's main parameter (line 10)
        agent.x = agent.aux_vars["phi"]

    def _communication(self, agent: Agent, network: P2PNetwork) -> None:
        """
        Communication phase (Algorithm 1, line 11).

        Transmit z_ij,k - 2 * rho * x_i,k+1 to each neighbor.
        """
        # Transmit z_i,k - 2 * rho * x_i,k+1 to each neighbor j in N_i
        for j in network.active_neighbors(agent):
            j_idx = agent.aux_vars["neighbor_to_idx"][j]
            message = agent.aux_vars["z_i"][j_idx] - 2 * self.penalty * agent.x
            network.send(agent, j, message)

    def _auxiliary_update(self, agent: Agent) -> None:
        """
        Auxiliary update phase (Algorithm 1, line 12).

        Update z_ij,k+1 according to equation (3b).
        """
        for j, msg in agent.messages.items():
            j_idx = agent.aux_vars["neighbor_to_idx"][j]
            z_update = (1 - self.alpha) * agent.aux_vars["z_i"][j_idx] - self.alpha * msg
            agent.aux_vars["z_i"][j_idx] = z_update
