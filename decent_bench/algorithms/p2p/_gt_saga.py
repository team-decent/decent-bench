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
class GT_SAGA(P2PAlgorithm):  # noqa: N801
    """
    Gradient Tracking with SAGA variance reduction :footcite:p:`Alg_GT_SAGA_2020` :footcite:p:`Alg_GT_SAGA_2022`.

    Args:
        iterations: Total number of iterations
        step_size: Step size for local updates
        x0: Initial parameters (optional)
        name: Algorithm name (default "GT-SAGA")

    Raises:
        TypeError: If any agent's cost function is not an instance of EmpiricalRiskCost.

    .. footbibliography::

    """

    iterations: int = 100
    step_size: float = 0.01
    x0: InitialStates = None  # Initial parameters (optional)
    name: str = "GT-SAGA"

    def __post_init__(self) -> None:
        """
        Validate parameters.

        Raises:
            ValueError: If any of the parameters are invalid (e.g., non-positive iterations, local_steps,
            step_size, penalty, or alpha).

        """
        if isinstance(self.step_size, float) and self.step_size <= 0:
            raise ValueError("step_size must be positive")

    def initialize(self, network: P2PNetwork) -> None:
        self.x0 = initial_states(self.x0, network)
        self.W = network.weights

        for i in network.agents():
            # Check that cost function supports SAGA
            if not isinstance(i.cost, EmpiricalRiskCost):
                raise TypeError("GT-SAGA only supports EmpiricalRiskCost instances.")

            # Initialize gradient table: z_{i,j}^0 = x_i^0 for all j
            z_grads = i.cost.gradient(self.x0[i], indices="all", reduction=None)

            # Initialize y_i^0 = 0_p and g_i^{-1} = 0_p
            y0 = iop.zeros_like(self.x0[i])
            g_minus1 = iop.zeros_like(self.x0[i])

            # Initialize auxiliary variables
            aux_vars = {
                "z_grads": z_grads,  # Gradient table z_{i,j}
                "y": y0,  # Gradient tracking variable y_i^0 = 0
                "g_old": g_minus1,  # Previous gradient estimator g_i^{-1} = 0
                "g": g_minus1,
            }
            i.initialize(x=self.x0[i], aux_vars=aux_vars)

    def step(self, network: P2PNetwork, _: int) -> None:
        # Step 1: Select random sample and update local stochastic gradient estimator
        for i in network.active_agents():
            self._update_gradient_estimator(i)

        # Step 2: Update gradient tracker
        # y_i^{k+1} = sum_{r=1}^n w_ir (y_r^k + g_r^k - g_r^{k-1})
        for i in network.active_agents():
            # Broadcast y_i + g_i - g_i^{-1}
            y_plus_delta_g = i.aux_vars["y"] + i.aux_vars["g"] - i.aux_vars["g_old"]
            i.aux_vars["y_plus_delta_g"] = y_plus_delta_g
            network.broadcast(i, y_plus_delta_g)

        for i in network.active_agents():
            self._update_gradient_tracker(i)

        # Step 3: Update local estimate of the solution
        # x_i^{k+1} = sum_{r=1}^n w_ir (x_r^k - alpha*y_r^{k+1})
        for i in network.active_agents():
            # Broadcast x_i - alpha*y_i to reduce communication
            x_minus_alpha_y = i.x - self.step_size * i.aux_vars["y"]
            i.aux_vars["x_minus_alpha_y"] = x_minus_alpha_y
            network.broadcast(i, x_minus_alpha_y)

        for i in network.active_agents():
            self._consensus_update(i)

        # Step 4: Update gradient table for a select samples
        for i in network.active_agents():
            self._update_gradient_table(i)

    def _update_gradient_estimator(self, agent: Agent) -> None:
        """
        Update local stochastic gradient estimator using SAGA variance reduction.

        Raises:
            TypeError: If the agent's cost function is not an instance of EmpiricalRiskCost.

        """
        if TYPE_CHECKING:
            if not isinstance(agent.cost, EmpiricalRiskCost):
                raise TypeError("GT-SAGA is only compatible with EmpiricalRiskCost.")

        # Store old g_i for gradient tracking update
        agent.aux_vars["g_old"] = agent.aux_vars["g"]

        # Compute grad f_{i,tau_i}(x_i^k), gradient at current point for selected sample
        grad_current = agent.cost.gradient(agent.x)
        batch_used = agent.cost.batch_used

        # Get grad f_{i,tau_i}(z_{i,tau_i}^k), gradient at stored point for selected sample
        z_grads = iop.mean(agent.aux_vars["z_grads"][batch_used], dim=0)

        # Compute (1/m) sum_{j=1}^m grad f_{i,j}(z_{i,j}^k), average of all gradients in table
        avg_table_grad = iop.mean(agent.aux_vars["z_grads"], dim=0)

        # Update SAGA gradient estimator
        # g_i^k = grad f_{i,tau_i}(x_i) - grad f_{i,tau_i}(z_{i,tau_i}) + (1/m) sum_{j=1}^m grad f_{i,j}(z_{i,j})
        agent.aux_vars["g"] = grad_current - z_grads + avg_table_grad

    def _update_gradient_tracker(self, agent: Agent) -> None:
        """Update local gradient tracker."""
        weighted_sum = self.W[agent, agent] * agent.aux_vars["y_plus_delta_g"]
        for j, y_plus_delta_g in agent.messages.items():
            weighted_sum += self.W[agent, j] * y_plus_delta_g
        agent.aux_vars["y"] = weighted_sum

    def _consensus_update(self, agent: Agent) -> None:
        """Update local estimate via consensus."""
        weighted_sum = self.W[agent, agent] * agent.aux_vars["x_minus_alpha_y"]
        for j, x_minus_alpha_y in agent.messages.items():
            weighted_sum += self.W[agent, j] * x_minus_alpha_y
        agent.x = weighted_sum

    def _update_gradient_table(self, agent: Agent) -> None:
        """
        Update gradient table for the selected sample.

        Raises:
            TypeError: If the agent's cost function is not an instance of EmpiricalRiskCost.

        """
        if TYPE_CHECKING:
            if not isinstance(agent.cost, EmpiricalRiskCost):
                raise TypeError("GT-SAGA is only compatible with EmpiricalRiskCost.")

        z_grads = agent.cost.gradient(agent.x, indices="batch", reduction=None)
        batch_used = agent.cost.batch_used

        # Update the gradient table entry for the selected sample
        agent.aux_vars["z_grads"][batch_used] = z_grads
        # All other entries remain unchanged (implicit)
