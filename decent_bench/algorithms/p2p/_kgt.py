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
class KGT(P2PAlgorithm):
    """
    K-GT: Gradient Sum Tracking algorithm :footcite:p:`Alg_K_GT`.

    Args:
        iterations: Total number of communication rounds (T)
        num_local_steps: Number of local gradient steps (K)
        step_size: Local step size (eta_c)
        aux_step_size: Communication step size (eta_s)
        x0: Initial parameters (optional)
        name: Algorithm name (default "K-GT")

    .. footbibliography::

    """

    iterations: int = 100  # Total number of communication rounds (T)
    num_local_steps: int = 5  # Number of local gradient steps (K)
    step_size: float = 0.01  # Local step size (eta_c)
    aux_step_size: float = 0.01  # Communication step size (eta_s)
    x0: InitialStates = None  # Initial parameters (optional)
    name: str = "K-GT"

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
        """
        Initialize agents with x_i^(0) and c_i^(0).

        Algorithm 1, line 2.
        """
        x0 = initial_states(self.x0, network)
        # Get mixing matrix weights
        self.W = network.weights

        for i in network.agents():
            # Initialize c_i^(0) according to line 2
            # In practice, we can initialize c_i^(0) = 0 (as noted in the paper)
            c_0 = iop.zeros_like(x0[i])

            i.initialize(
                x=x0[i],
                aux_vars={
                    "c": c_0,
                    "x_before_local": x0[i],
                    "z_i": x0[i],
                },
            )

    def step(self, network: P2PNetwork, _: int) -> None:
        # Main algorithm loop (lines 4-12)
        # Step 1: Local training phase (lines 5-7)
        for i in network.active_agents():
            self._local_training(i)

        # Step 2: Compute z_i and store in aux_vars
        multiplier = self.num_local_steps * self.aux_step_size * self.step_size
        for i in network.active_agents():
            # Compute z_i^(t) = (1/K eta_c)(x_i^(t) - x_i^(t+K))
            i.aux_vars["z_i"] = (i.aux_vars["x_before_local"] - i.x) / (self.num_local_steps * self.step_size)
            msg = i.aux_vars["x_before_local"] - multiplier * i.aux_vars["z_i"]
            i.aux_vars["msg"] = msg
            # Step 3: Communication phase
            message = iop.stack([msg, i.aux_vars["z_i"]])
            network.broadcast(i, message)

        # Step 5: Update tracking variable and model parameters (lines 9-10)
        for i in network.active_agents():
            self._update_tracking_and_params(i)

    def _local_training(self, agent: Agent) -> None:
        """
        Perform K local gradient steps.

        Algorithm 1, lines 5-7.
        """
        # Store x_i^(t) before local steps for z_i computation
        agent.aux_vars["x_before_local"] = agent.x
        x_k = iop.copy(agent.x)

        # Perform K local gradient steps (line 6)
        # x_i^(t)+k+1 = x_i^(t)+k - eta_c(grad F_i(x_i^(t)+k; xi_i^(t)+k) + c_i^(t))
        for _ in range(self.num_local_steps):
            gradient = agent.cost.gradient(x_k)
            x_k -= self.step_size * (gradient + agent.aux_vars["c"])
        agent.x = x_k

    def _update_tracking_and_params(
        self,
        agent: Agent,
    ) -> None:
        """
        Update tracking variable c_i and model parameters x_i.

        Algorithm 1, lines 9-10.
        """
        # Get z_i (already computed)
        z_i = agent.aux_vars["z_i"]

        # Line 9: Update tracking variable
        # c_i^(t+1) = c_i^(t) - z_i^(t) + ∑_j w_ij z_j^(t)
        # Line 10: Update model parameters
        weighted_neighbor_z = self.W[agent, agent] * z_i
        weighted_sum = self.W[agent, agent] * agent.aux_vars["msg"]
        for j, msg in agent.messages.items():
            weighted_neighbor_z += self.W[agent, j] * msg[1]
            weighted_sum += self.W[agent, j] * msg[0]
        agent.aux_vars["c"] = agent.aux_vars["c"] - z_i + weighted_neighbor_z
        agent.x = weighted_sum
