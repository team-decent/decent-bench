from dataclasses import dataclass

import decent_bench.utils.interoperability as iop
from decent_bench.agents import Agent
from decent_bench.algorithms.utils import initial_states
from decent_bench.networks import P2PNetwork
from decent_bench.utils._tags import tags
from decent_bench.utils.types import InitialStates

from ._p2p_algorithm import P2PAlgorithm


@tags("peer-to-peer", "gradient-based")
@dataclass(eq=False)
class LED(P2PAlgorithm):
    """
    Local Exact-Diffusion (LED) algorithm :footcite:p:`Alg_LED`.

    Args:
        iterations: Total number of communication rounds (r)
        num_local_steps: Number of local updates (tau)
        step_size: Step size alpha for gradient steps
        aux_step_size: Step size beta for dual variable
        x0: Initial parameters (optional)
        name: Algorithm name (default "LED")

    .. footbibliography::

    """

    iterations: int = 100  # Total number of communication rounds (r)
    num_local_steps: int = 5  # Number of local updates (tau)
    step_size: float = 0.01  # Step size alpha for gradient steps
    aux_step_size: float = 0.01  # Step size beta for dual variable
    x0: InitialStates = None  # Initial parameters (optional)
    name: str = "LED"

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

    def initialize(self, network: P2PNetwork) -> None:
        """Initialize agents with x_i^0, y_i^0, and phi_i,0^r."""
        x0 = initial_states(self.x0, network)
        self.W = network.weights

        for i in network.agents():
            # Initialize y_i^0 = 0 (simplified initialization)
            y_0 = iop.zeros_like(x0[i])

            # Initialize auxiliary variables
            aux_vars = {
                "y": y_0,  # Dual variable y_i^r
                "phi": x0[i],  # phi_i,tau^r (to be broadcasted)
            }

            i.initialize(
                x=x0[i],
                aux_vars=aux_vars,
            )

    def step(self, network: P2PNetwork, _: int) -> None:
        # Step 1: Local primal updates (tau steps)
        for i in network.active_agents():
            self._local_primal_updates(i)

        # Step 2: Diffusion (communication and mixing)
        for i in network.active_agents():
            network.broadcast(i, i.aux_vars["phi"])

        for i in network.active_agents():
            self._diffusion(i)

        # Step 3: Local dual update
        for i in network.active_agents():
            self._local_dual_update(i)

    def _local_primal_updates(self, agent: Agent) -> None:
        """
        Step 1: Local primal updates (tau steps).

        Algorithm 1, line 1:
        """
        # Set phi_i,0^r = x_i^r (line 1)
        agent.aux_vars["phi"] = iop.copy(agent.x)

        # Perform tau local updates (Equation 2a)
        for _ in range(self.num_local_steps):
            gradient = agent.cost.gradient(agent.aux_vars["phi"])
            agent.aux_vars["phi"] -= self.step_size * gradient + self.aux_step_size * agent.aux_vars["y"]

    def _diffusion(self, agent: Agent) -> None:
        """
        Step 2: Diffusion.

        Algorithm 1, line 2:
        """
        weighted_sum = self.W[agent, agent] * agent.aux_vars["phi"]
        for j, phi_j in agent.messages.items():
            weighted_sum += self.W[agent, j] * phi_j
        agent.x = weighted_sum

    def _local_dual_update(self, agent: Agent) -> None:
        """
        Step 3: Local dual update.

        Algorithm 1, line 3:

        Update the dual variable for exact tracking (Equation 2c).
        """
        agent.aux_vars["y"] += agent.aux_vars["phi"] - agent.x
