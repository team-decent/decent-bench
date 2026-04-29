import random
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
class ProxSkip(P2PAlgorithm):
    """
    Proximal Skip with local gradient steps :footcite:p:`Alg_Prox_Skip`.

    Args:
        iterations: Total number of iterations (T)
        step_size: Step size alpha > 0 for primal updates
        aux_step_size: Step size beta > 0 for dual updates
        comm_probability: Communication probability 0 < p <= 1 for skipping communication
        chi: chi >= 1, averaging weight parameter for weighted averaging during communication
        x0: Initial parameters (optional)
        name: Algorithm name (default "ProxSkip")

    .. footbibliography::

    """

    iterations: int = 100  # Total number of iterations (T)
    step_size: float = 0.01  # Step size alpha > 0 for primal updates
    aux_step_size: float = 0.01  # Step size beta > 0 for dual updates
    comm_probability: float = 0.7  # Communication probability 0 < p <= 1
    chi: float = 1.0  # chi >= 1, averaging weight parameter
    x0: InitialStates = None  # Initial parameters (optional)
    name: str = "ProxSkip"

    def __post_init__(self) -> None:
        """
        Validate parameters.

        Raises:
            ValueError: If any of the parameters are invalid (e.g., non-positive iterations, local_steps,
            step_size, penalty, or alpha).

        """
        if isinstance(self.step_size, float) and self.step_size <= 0:
            raise ValueError("step_size must be positive")
        if isinstance(self.aux_step_size, float) and self.aux_step_size <= 0:
            raise ValueError("aux_step_size must be positive")
        if not 0 < self.comm_probability <= 1:
            raise ValueError("comm_probability must be in (0, 1]")
        if self.chi < 1:
            raise ValueError("chi must be >= 1")

    def initialize(self, network: P2PNetwork) -> None:
        """
        Initialize agents with x_i^0, y_i^0 = 0, and compute weights W_a.

        Algorithm 1, line 1:
        """
        x0 = initial_states(self.x0, network)
        agents = network.agents()

        # Compute weights W_a = I - 1/(2χ)(I - W)
        n = len(agents)
        W = network.weights  # noqa: N806
        I = iop.eye(n=n, framework=agents[0].cost.framework, device=agents[0].cost.device)  # noqa: E741, N806
        self.W_a = I - (1.0 / (2.0 * self.chi)) * (I - W)

        for i in agents:
            # Initialize y_i^0 = 0 (dual variable)
            y_0 = iop.zeros_like(x0[i])

            # Initialize auxiliary variables
            aux_vars = {
                "y": y_0,  # Dual/control variable y_i^t
                "z": x0[i],  # Prediction variable z_i^t
            }
            i.initialize(x=x0[i], aux_vars=aux_vars)

    def step(self, network: P2PNetwork, _: int) -> None:
        # Main algorithm loop (line 3)
        # Step 1: Sample stochastic gradient and compute prediction (lines 4-5)
        for i in network.active_agents():
            self._compute_prediction(i)

        # Step 2: Flip coins to determine communication (line 2)
        # theta_k ~ Bernoulli(p), with P(theta_k = 1) = p
        theta_k = random.random() < self.comm_probability

        # Step 3: Communication and updates (lines 6-11)
        if theta_k:  # theta_k = 1 (communicate with probability p)
            for i in network.active_agents():
                # Line 7: Broadcast z_i^t for weighted averaging
                network.broadcast(i, i.aux_vars["z"])

        # Update based on communication decision
        for i in network.active_agents():
            if theta_k:  # Line 7-8: communicate and update
                self._communication_update(i)
            else:  # Line 10: skip communication
                self._no_communication_update(i)

    def _compute_prediction(self, agent: Agent) -> None:
        """
        Sample gradient and update prediction variable.

        Algorithm 1, lines 4-5:
        """
        # Sample stochastic gradient g_i^t = grad F_i(x_i^t, xi_i^t)
        gradient = agent.cost.gradient(agent.x)

        # Update prediction: z_i^t = x_i^t - alpha * g_i^t - y_i^t
        agent.aux_vars["z"] = agent.x - self.step_size * gradient - agent.aux_vars["y"]

    def _communication_update(self, agent: Agent) -> None:
        """
        Communication and update when θ_i = 1.

        Algorithm 1, lines 7-8:
        """
        # Compute weighted average: x_i^{t+1} = sum_{j=1}^n W_ij z_j^t
        # In practice, we only communicate with neighbors, so:
        # x_i^{t+1} = sum_{j in N_i} W_a[i,j] z_j^t
        weighted_sum = self.W_a[agent, agent] * agent.aux_vars["z"]
        for j, z_j in agent.messages.items():
            weighted_sum += self.W_a[agent, j] * z_j
        # Update primal: x_i^{t+1} = weighted average
        agent.x = weighted_sum

        # Update dual: y_i^{t+1} = y_i^t + beta * (z_i^t - x_i^{t+1})
        agent.aux_vars["y"] += self.aux_step_size * (agent.aux_vars["z"] - agent.x)

    def _no_communication_update(self, agent: Agent) -> None:
        """
        Skip communication when θ_i = 0.

        Algorithm 1, line 10:
        """
        # Update primal: x_i^{t+1} = z_i^t (use prediction)
        agent.x = agent.aux_vars["z"]

        # Dual variable unchanged: y_i^{t+1} = y_i^t
        # (no explicit update needed, aux_vars["y"] stays the same)
