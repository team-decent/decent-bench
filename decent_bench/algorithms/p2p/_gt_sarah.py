from dataclasses import dataclass
from typing import TYPE_CHECKING

import decent_bench.utils.interoperability as iop
from decent_bench.agents import Agent
from decent_bench.algorithms.utils import initial_states
from decent_bench.costs import EmpiricalRiskCost
from decent_bench.networks import P2PNetwork
from decent_bench.utils._tags import tags
from decent_bench.utils.types import InitialStates

from ._p2p_algorithm import P2PAlgorithm


@tags("peer-to-peer", "gradient-tracking")
@dataclass(eq=False)
class GT_SARAH(P2PAlgorithm):  # noqa: N801
    """
    GT-SARAH: Gradient Tracking with SARAH variance reduction :footcite:p:`Alg_GT_SARAH`.

    Args:
        iterations: Total number of outer loops (S)
        num_local_steps: Number of inner loop iterations (q)
        step_size: Step size (alpha) for updates
        x0: Initial parameters (optional)
        name: Algorithm name (default "GT-SARAH")

    Raises:
        TypeError: If any agent's cost function is not an instance of EmpiricalRiskCost.

    .. footbibliography::

    """

    iterations: int = 100  # S: number of outer loops
    num_local_steps: int = 5  # q: number of inner loop iterations
    step_size: float = 0.01  # alpha: step size
    x0: InitialStates = None  # Initial parameters (optional)
    name: str = "GT-SARAH"

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

    def initialize(self, network: P2PNetwork) -> None:
        """
        Initialize agents with x_i^{0,1}, y_i^{0,1}, v_i^{-1,1}.

        Raises:
            TypeError: If any agent's cost function is not an instance of EmpiricalRiskCost.

        """
        self.x0 = initial_states(self.x0, network)
        self.W = network.weights

        for i in network.agents():
            # Check that cost function supports variance reduction
            if not isinstance(i.cost, EmpiricalRiskCost):
                raise TypeError("GT-SARAH only supports EmpiricalRiskCost instances.")

            # Initialize y_i^{0,1} = 0 and v_i^{-1,1} = 0
            y0 = iop.zeros_like(self.x0[i])
            v_minus1 = iop.zeros_like(self.x0[i])

            # Initialize auxiliary variables
            aux_vars = {
                "y": y0,  # Gradient tracking variable y_i^{0,1} = 0
                "v": v_minus1,  # Current SARAH estimator
                "v_prev": v_minus1,  # v_i^{-1,1} = 0 (for outer loop tracking)
                "x_prev": self.x0[i],  # Store x_{t-1} for SARAH
            }
            # Estimate received messages using agent's own initial values
            i.initialize(
                x=self.x0[i],
                aux_vars=aux_vars,
            )

    def step(self, network: P2PNetwork, _: int) -> None:
        # Step 1: Compute full gradient (batch gradient computation)
        for i in network.active_agents():
            self._compute_batch_grad(i)

        # Step 2: Update gradient tracker
        for i in network.active_agents():
            network.broadcast(i, i.aux_vars["y"])

        for i in network.active_agents():
            self._update_gradient_tracker(i)

        # Step 3: Update state
        for i in network.active_agents():
            network.broadcast(i, i.x)

        for i in network.active_agents():
            self._state_update(i)

        self._inner_loop(network)

    def _compute_batch_grad(self, agent: Agent) -> None:
        """
        Compute full gradient at the beginning of each outer loop.

        Algorithm 2.1, line 2:
        """
        agent.aux_vars["v_prev"] = agent.aux_vars["v"]
        grad = agent.cost.gradient(agent.x, indices="all")

        # Update v_i^{0,s} = grad f_i(x_i^{0,s})
        agent.aux_vars["v"] = grad

    def _update_sarah_estimator(self, agent: Agent) -> None:
        """
        Update SARAH variance-reduced gradient estimator.

        Algorithm 2.1, line 8.

        Raises:
            TypeError: If the agent's cost function is not an instance of EmpiricalRiskCost.

        """
        if TYPE_CHECKING:
            if not isinstance(agent.cost, EmpiricalRiskCost):
                raise TypeError("GT-SAGA is only compatible with EmpiricalRiskCost.")

        # Store previous inner loop gradient for tracking update
        agent.aux_vars["v_prev"] = agent.aux_vars["v"]

        # Compute (1/B) sum_{l=1}^B grad f_{i,tau_l}(x_i^{t,s})
        grad_current = agent.cost.gradient(agent.x)
        batch_used = agent.cost.batch_used

        # Compute (1/B) sum_{l=1}^B grad f_{i,tau_l}(x_i^{t-1,s})
        grad_prev = agent.cost.gradient(agent.aux_vars["x_prev"], indices=batch_used)

        # SARAH update: v_i^{t,s} = (grad f_i(x_i, xi) - grad f_i(x_prev, xi)) + v_i^{t-1,s}
        agent.aux_vars["v"] = grad_current - grad_prev + agent.aux_vars["v_prev"]

    def _update_gradient_tracker(self, agent: Agent) -> None:
        """
        Update gradient tracker at the beginning of outer loop.

        Algorithm 2.1, line 3 and 9.

        """
        weighted_sum = self.W[agent, agent] * agent.aux_vars["y"]
        for j, y in agent.messages.items():
            weighted_sum += self.W[agent, j] * y
        agent.aux_vars["y"] = weighted_sum + agent.aux_vars["v"] - agent.aux_vars["v_prev"]

    def _state_update(self, agent: Agent) -> None:
        """
        Update local estimate via consensus.

        Algorithm 2.1, lines 4 and 10:

        """
        agent.aux_vars["x_prev"] = agent.x
        weighted_sum = self.W[agent, agent] * agent.x
        for j, x in agent.messages.items():
            weighted_sum += self.W[agent, j] * x
        agent.x = weighted_sum - self.step_size * agent.aux_vars["y"]

    def _inner_loop(self, network: P2PNetwork) -> None:
        """
        Inner loop of GT-SARAH.

        Algorithm 2.1, lines 7-10.
        """
        for _ in range(self.num_local_steps):
            # Step 4: SARAH variance reduction
            for i in network.active_agents():
                self._update_sarah_estimator(i)  # line 8

            # Step 5: Update gradient tracker (inner loop)
            for i in network.active_agents():
                network.broadcast(i, i.aux_vars["y"])

            for i in network.active_agents():
                self._update_gradient_tracker(i)  # line 9

            # Step 6: Update state (inner loop)
            for i in network.active_agents():
                network.broadcast(i, i.x)

            for i in network.active_agents():
                self._state_update(i)
