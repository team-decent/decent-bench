from dataclasses import dataclass
from typing import TYPE_CHECKING

import decent_bench.utils.interoperability as iop
from decent_bench.agents import Agent
from decent_bench.algorithms.utils import initial_states
from decent_bench.networks import P2PNetwork
from decent_bench.utils._tags import tags
from decent_bench.utils.types import InitialStates

from ._p2p_algorithm import P2PAlgorithm

if TYPE_CHECKING:
    from decent_bench.utils.array import Array


@tags("peer-to-peer", "gradient-based")
@dataclass(eq=False)
class DiNNO(P2PAlgorithm):
    r"""
    Distributed Neural Network Optimization (DiNNO) algorithm :footcite:p:`Alg_DiNNO`.

    Each iteration, each agent approximately optimizes an augmented Lagrangian function which
    is then communicated to its neighbors in order to update the dual variables. This is
    then repeated for a number of iterations.

    Args:
        iterations: Total number of outer iterations (K)
        step_size: Step size for primal updates
        num_local_steps: Number of inner iterations (B) for approximate primal update
        penalty: Penalty parameter (rho) for augmented Lagrangian
        x0: Initial parameters (optional)
        name: Algorithm name (default "DiNNO")

    .. footbibliography::

    """

    iterations: int = 100  # Total number of outer iterations (K)
    step_size: float = 0.01
    num_local_steps: int = 5  # Number of inner iterations (B) for approximate primal update
    penalty: float = 0.5  # Penalty parameter (rho) for augmented Lagrangian
    x0: InitialStates = None  # Initial parameters (optional)
    name: str = "DiNNO"

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
        if self.penalty <= 0:
            raise ValueError("penalty must be positive")

    def initialize(self, network: P2PNetwork) -> None:
        # Initialize agents (lines 2-5)
        self.x0 = initial_states(self.x0, network)

        for i in network.agents():
            # Initialize dual variable p_i = theta (line 3)
            p_0 = iop.zeros_like(self.x0[i])

            i.initialize(
                x=self.x0[i],  # theta_i^theta = theta_initial (line 4)
                aux_vars={"p": p_0},
            )

    def step(self, network: P2PNetwork, _: int) -> None:
        # Main optimization loop (line 7)
        # Step 1: Communication - send θ_i^k to neighbors (line 8)
        for i in network.active_agents():
            network.broadcast(i, i.x)

        # Step 2: Dual variable update (line 10) - Equation (4a)
        for i in network.active_agents():
            self._auxiliary_update(i)

        # Step 3: Approximate primal update (lines 11-15) - Equation (4b)
        for i in network.active_agents():
            self._local_training(i)

    def _auxiliary_update(self, agent: Agent) -> None:
        # p_i^(k+1) = p_i^k + rho * sum_{j in N_i}(theta_i^k - theta_j^k)
        s = None
        for val in agent.messages.values():
            if s is None:
                s = val
            else:
                s += val
        if s is not None:
            consensus_error = agent.x * len(agent.messages) - s
            agent.aux_vars["p"] += self.penalty * consensus_error

    def _local_training(self, agent: Agent) -> None:
        # Initialize psi^0 = theta_i^k (line 11)
        psi = iop.copy(agent.x)

        neighbor_thetas_sum: Array | None = None
        for val in agent.messages.values():
            if neighbor_thetas_sum is None:
                neighbor_thetas_sum = val
            else:
                neighbor_thetas_sum += val
        if neighbor_thetas_sum is not None:
            neighbor_thetas_sum /= 2.0

        # Approximate the primal update with B iterations (lines 12-14)
        for _ in range(self.num_local_steps):
            # Term 1: grad l(psi; D_i)
            grad_loss = agent.cost.gradient(psi)

            # Term 2: grad(theta^T p_i^(k+1)) = p_i^(k+1)
            grad_dual = agent.aux_vars["p"]

            # Term 3: grad[rho sum_{j in N_i} ||theta - (theta_i^k + theta_j^k)/2||^2]
            #       = 2 rho sum_{j in N_i} (theta - (theta_i^k + theta_j^k)/2)
            if neighbor_thetas_sum is not None:
                consensus_term = (psi - agent.x / 2) * len(agent.messages) - neighbor_thetas_sum
                # Note: factor of 2 from derivative of squared norm
                grad_consensus: Array | float = 2.0 * self.penalty * consensus_term
            else:
                grad_consensus = 0.0

            # Gradient step: psi^(tau+1) = psi^tau - step_size * grad L_augmented
            total_gradient = grad_loss + grad_dual + grad_consensus
            psi -= self.step_size * total_gradient

        # Update primal variable theta_i^(k+1) = psi^B (line 15)
        agent.x = psi
