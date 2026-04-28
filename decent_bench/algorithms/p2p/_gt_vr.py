import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from decent_bench.agents import Agent
from decent_bench.algorithms.utils import initial_states
from decent_bench.costs import EmpiricalRiskCost
from decent_bench.networks import P2PNetwork
from decent_bench.utils._tags import tags
from decent_bench.utils.types import InitialStates

from ._p2p_algorithm import P2PAlgorithm


@tags("peer-to-peer", "gradient-tracking")
@dataclass(eq=False)
class GT_VR(P2PAlgorithm):  # noqa: N801
    """
    GT-VR: Gradient Tracking with Variance Reduction algorithm :footcite:p:`Alg_GT_VR`.

    Args:
        iterations: Total number of iterations
        step_size: Step size for primal updates
        snapshot_prob: Probability of performing a snapshot update (P in the paper)
        x0: Initial parameters (optional)
        name: Algorithm name (default "GT-VR")

    Raises:
        TypeError: If any agent's cost function is not an instance of EmpiricalRiskCost.

    .. footbibliography::

    """

    iterations: int = 100
    step_size: float = 0.01
    snapshot_prob: float = 0.3  # P in the algorithm
    x0: InitialStates = None  # Initial parameters (optional)
    name: str = "GT-VR"

    def __post_init__(self) -> None:
        """
        Validate parameters.

        Raises:
            ValueError: If any of the parameters are invalid (e.g., non-positive iterations, local_steps,
            step_size, penalty, or alpha).

        """
        if isinstance(self.step_size, float) and self.step_size <= 0:
            raise ValueError("step_size must be positive")
        if not 0 < self.snapshot_prob <= 1:
            raise ValueError("snapshot_prob must be in (0, 1]")

    def initialize(self, network: P2PNetwork) -> None:
        """
        Initialize agents.

        Algorithm 1, line 1

        Raises:
            TypeError: If any agent's cost function is not an instance of EmpiricalRiskCost, since GT-VR relies on
                variance reduction techniques that require access to individual sample gradients. Using GT-VR with
                incompatible cost functions may lead to errors or undefined behavior.

        """
        self.x0 = initial_states(self.x0, network)
        self.W = network.weights

        for i in network.agents():
            # Check that cost function supports variance reduction
            if not isinstance(i.cost, EmpiricalRiskCost):
                raise TypeError("GT-VR only supports EmpiricalRiskCost instances.")

            # Compute full gradient at initialization: grad f_i(x_i^1)
            full_grad = i.cost.gradient(self.x0[i], indices="all")

            # Initialize auxiliary variables according to line 1
            aux_vars = {
                "tau": (self.x0[i]),  # tau_i^1 = x_i^1 (for snapshot updates)
                "full_grad_tau": full_grad,  # grad f_i(tau_i) - cached to avoid recomputation
                "y": full_grad,  # y_i^1 = grad f_i(x_i^1)
                "v": full_grad,  # v_i^1 = grad f_i(x_i^1)
                "v_old": full_grad,  # Store v_i^k for gradient tracking update
            }
            i.initialize(
                x=self.x0[i],
                aux_vars=aux_vars,
            )

    def step(self, network: P2PNetwork, _: int) -> None:
        # Main algorithm loop (line 2)
        for i in network.active_agents():
            x_minus_eta_y = i.x - self.step_size * i.aux_vars["y"]
            i.aux_vars["x_minus_eta_y"] = x_minus_eta_y
            network.broadcast(i, x_minus_eta_y)

        # Step 1: Update local estimate of the solution (line 3)
        for i in network.active_agents():
            self._consensus_update(i)

        # Step 2: Probabilistic snapshot update (line 4)
        # Select l_i^{k+1} ~ Bernoulli(P)
        for i in network.active_agents():
            if random.random() < self.snapshot_prob:  # l_i^{k+1} = 1
                # Update: tau_i^{k+1} = x_i^{k+1} and recompute full gradient
                self._snapshot_update(i)

        # Step 3: Select batch and update local gradient estimator (lines 5-6)
        for i in network.active_agents():
            self._update_gradient_estimator(i)

        # We broadcast y_i + v_i - v_old to reduce communication
        for i in network.active_agents():
            y_plus_delta_v = i.aux_vars["y"] + i.aux_vars["v"] - i.aux_vars["v_old"]
            i.aux_vars["y_plus_delta_v"] = y_plus_delta_v
            network.broadcast(i, y_plus_delta_v)

        # Step 4: Update gradient tracker (line 7)
        for i in network.active_agents():
            self._update_gradient_tracker(i)

    def _consensus_update(self, agent: Agent) -> None:
        """
        Update local estimate via consensus.

        Algorithm 1, line 3.

        """
        weighted_sum = self.W[agent, agent] * agent.aux_vars["x_minus_eta_y"]
        for j, x_minus_eta_y in agent.messages.items():
            weighted_sum += self.W[agent, j] * x_minus_eta_y
        agent.x = weighted_sum

    def _snapshot_update(self, agent: Agent) -> None:
        """
        Update snapshot point when l_i^{k+1} = 1.

        Algorithm 1, line 4.

        """
        agent.aux_vars["tau"] = agent.x

        # Compute and cache the full gradient at the new snapshot point
        full_grad_tau = agent.cost.gradient(agent.aux_vars["tau"], indices="all")
        agent.aux_vars["full_grad_tau"] = full_grad_tau

    def _update_gradient_estimator(self, agent: Agent) -> None:
        """
        Update local stochastic gradient estimator with variance reduction.

        Algorithm 1, lines 5-6:

        This implements the variance reduction technique (Equation 3)

        Raises:
            TypeError: If the agent's cost is not an instance of EmpiricalRiskCost.

        """
        if TYPE_CHECKING:
            if not isinstance(agent.cost, EmpiricalRiskCost):
                raise TypeError("GT-VR is only compatible with EmpiricalRiskCost.")

        # Store old v_i for gradient tracking update
        agent.aux_vars["v_old"] = agent.aux_vars["v"]

        # Select s_i^{k+1} uniformly at random (this is done by the cost function)
        # Compute stochastic gradient at current point: grad f_{i,s_i}(x_i^{k+1})
        grad_current = agent.cost.gradient(agent.x)
        batch_indices = agent.cost.batch_used

        # Compute stochastic gradient at snapshot point: grad f_{i,s_i}(tau_i^{k+1})
        grad_snapshot = agent.cost.gradient(agent.aux_vars["tau"], indices=batch_indices)

        # Use cached full gradient at snapshot point: grad f_i(tau_i^{k+1})
        full_grad_snapshot = agent.aux_vars["full_grad_tau"]

        # Update variance-reduced gradient estimator (Equation 3)
        # v_i^{k+1} = grad f_{i,s_i}(x_i) - grad f_{i,s_i}(tau_i) + grad f_i(tau_i)
        agent.aux_vars["v"] = grad_current - grad_snapshot + full_grad_snapshot

    def _update_gradient_tracker(self, agent: Agent) -> None:
        """
        Update local gradient tracker.

        Algorithm 1, line 7:

        Note:
            We receive y_r + v_r - v_r_old directly to reduce communication.

        """
        weighted_sum = self.W[agent, agent] * agent.aux_vars["y_plus_delta_v"]
        for j, y_plus_delta_v in agent.messages.items():
            weighted_sum += self.W[agent, j] * y_plus_delta_v
        agent.aux_vars["y"] = weighted_sum
