from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from rich.progress import track

import decent_bench.algorithms.algorithm_helpers as alg_helpers
import decent_bench.utils.interoperability as iop
from decent_bench.agents import Agent
from decent_bench.costs import PyTorchCost
from decent_bench.distributed_algorithms import Algorithm
from decent_bench.networks import P2PNetwork
from decent_bench.utils.logger import LOGGER

if TYPE_CHECKING:
    from decent_bench.utils.array import Array


@dataclass(eq=False)
class LT_ADMM_EMA(Algorithm):  # noqa: N801
    """
    Local Training ADMM with exponential moving averages algorithm for distributed optimization.

    Args:
        iterations: Total number of communication rounds (K)
        local_steps: Number of local training steps (tau)
        step_size: Local step size (gamma), can be a constant or a function of iteration
        penalty: Penalty parameter (rho)
        alpha: Relaxation parameter (alpha)
        ema_factor: Exponential moving average factor, ema_factor * old + (1 - ema_factor) * new
        set_x_to_ema: Whether to set final x to EMA values in finalize()
        use_z_ema: Whether to use EMA for z_ij,k updates in auxiliary update
        use_torch_optim: Whether to use PyTorch optimizers for local training
        x0: Initial parameters (optional)
        name: Algorithm name (default "LT-ADMM-EMA")

    """

    iterations: int = 100  # Total number of communication rounds (K)
    local_steps: int = 5  # Number of local training steps (tau)
    step_size: float | Callable[[int], float] = 0.01  # Local step size (gamma)
    penalty: float = 1.0  # Penalty parameter (rho)
    alpha: float = 0.5  # Relaxation parameter (alpha)
    ema_factor: float = (
        0.9  # Exponential moving average factor, ema_factor * old + (1 - ema_factor) * new
    )
    set_x_to_ema: bool = True  # Whether to set final x to EMA values
    use_z_ema: bool = True  # Whether to use EMA for z_ij,k updates
    use_torch_optim: bool = True  # Whether to use PyTorch optimizers for local training
    x0: "Array | None" = None  # Initial parameters (optional)
    name: str = "LT-ADMM-EMA"

    def __post_init__(self) -> None:
        """
        Validate parameters.

        Raises:
            ValueError: If any of the parameters are invalid (e.g., non-positive iterations, local_steps, step_size,
            penalty, or alpha).

        """
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.local_steps <= 0:
            raise ValueError("local_steps must be positive")
        if isinstance(self.step_size, float) and self.step_size <= 0:
            raise ValueError("step_size must be positive")
        if callable(self.step_size):
            test_step_size = [self.step_size(k) for k in range(self.iterations)]
            if any(s <= 0 for s in test_step_size):
                raise ValueError(
                    "step_size function must return positive values for all iterations"
                )
        if self.penalty <= 0:
            raise ValueError("penalty must be positive")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")

    def initialize(self, network: P2PNetwork) -> None:  # noqa: D102
        self.x0 = alg_helpers.zero_initialization(self.x0, network)
        self.optimizers: dict[Agent, torch.optim.Optimizer] = {}
        self.schedulers: dict[Agent, torch.optim.lr_scheduler.LambdaLR] = {}

        # Initialize agents with auxiliary variables
        for i in track(
            network.agents(),
            description="Initializing agents...",
            transient=True,
        ):
            if not isinstance(i.cost, PyTorchCost):
                raise TypeError(
                    f"LT-ADMM-EMA requires PyTorchCost, but agent {i} has cost of type {type(i.cost)}"
                )

            neighbors = network.neighbors(i)
            z_i = iop.zeros(
                (len(neighbors), *iop.shape(self.x0)),
                framework=i.cost.framework,
                device=i.cost.device,
            )
            neighbor_to_idx = {}

            for idx, j in enumerate(neighbors):
                z_i[idx] = iop.copy(self.x0)
                neighbor_to_idx[j] = idx

            if self.use_torch_optim:
                self.optimizers[i] = torch.optim.Adam(
                    i.cost.model.parameters(),
                    lr=(
                        self.step_size(0)
                        if callable(self.step_size)
                        else self.step_size
                    ),
                )
                if callable(self.step_size):
                    self.schedulers[i] = torch.optim.lr_scheduler.LambdaLR(
                        self.optimizers[i], lr_lambda=self.step_size
                    )

            aux_vars = {
                "phi": self.x0,  # phi_i,k - model parameters
                "phi_ema": self.x0,  # Exponential moving average of phi_i,k
                "z_i": z_i,  # z_ij,k+1 - auxiliary consensus variable
                "neighbor_to_idx": neighbor_to_idx,
            }
            i.initialize(
                x=self.x0,
                aux_vars=aux_vars,
                received_msgs=dict.fromkeys(neighbors, self.x0),
            )

        LOGGER.info("Initialization complete")

    def step(self, network: P2PNetwork, iteration: int) -> None:  # noqa: D102
        step_size = (
            self.step_size(iteration) if callable(self.step_size) else self.step_size
        )

        # Step 1: Local training phase
        for i in network.active_agents(iteration):
            self._local_training(i, network, step_size)

        # Step 2: Communication phase
        for i in network.active_agents(iteration):
            self._communication(i, network)

        # Step 3: Auxiliary update phase
        for i in network.active_agents(iteration):
            self._auxiliary_update(i, network)

    def finalize(self, network: P2PNetwork) -> None:  # noqa: D102
        # Set final parameters to EMA values
        if self.set_x_to_ema:
            for agent in network.agents():
                agent.x = iop.copy(agent.aux_vars["phi_ema"])

    def _local_training(
        self,
        agent: Agent,
        network: P2PNetwork,
        step_size: float,
    ) -> None:
        """
        Perform local training steps.

        Updates phi_i,k and gradient estimators r_i,h,k.

        Raises:
            TypeError: If the agent's cost is not a PyTorchCost, since this implementation relies on
                PyTorch for local optimization.

        """
        if not isinstance(agent.cost, PyTorchCost):
            raise TypeError(
                f"LT-ADMM-EMA requires PyTorchCost, but agent {agent} has cost of type {type(agent.cost)}"
            )

        neighbors = network.neighbors(agent)

        agent.aux_vars["phi"] = iop.copy(agent.x)
        z_sum = iop.sum(agent.aux_vars["z_i"], dim=0)

        if self.use_torch_optim:
            # Set model parameters if using PyTorch optimizers
            agent.cost._set_model_parameters(agent.aux_vars["phi"])
            agent.cost.model.train()  # Set model to training mode

        for _ in range(self.local_steps):
            if self.use_torch_optim:
                # Use PyTorch optimizer for local training
                self.optimizers[agent].zero_grad()
                batch_x, batch_y = agent.cost._get_batch_data("batch")
                pred_y = agent.cost.model(batch_x)
                loss = agent.cost.loss_fn(pred_y, batch_y)
                loss.backward()
                self.optimizers[agent].step()
            else:
                # Manual gradient step
                current_gradient = agent.cost.gradient(agent.aux_vars["phi"])
                step = (
                    current_gradient
                    + self.penalty * len(neighbors) * agent.aux_vars["phi"]
                    - z_sum
                )
                agent.aux_vars["phi"] -= step_size * step

        if agent in self.schedulers:
            self.schedulers[agent].step()

        agent.x = agent.aux_vars["phi"]
        agent.aux_vars["phi_ema"] = (
            self.ema_factor * agent.aux_vars["phi_ema"]
            + (1 - self.ema_factor) * agent.aux_vars["phi"]
        )

    def _communication(self, agent: Agent, network: P2PNetwork) -> None:
        """
        Communication phase (Algorithm 1, line 11).

        Transmit z_ij,k - 2 * rho * x_i,k+1 to each neighbor.
        """
        for j in network.neighbors(agent):
            j_idx = agent.aux_vars["neighbor_to_idx"][j]
            message = agent.aux_vars["z_i"][j_idx] - 2 * self.penalty * agent.x
            network.send(agent, j, message)

    def _auxiliary_update(self, agent: Agent, network: P2PNetwork) -> None:
        """
        Auxiliary update phase (Algorithm 1, line 12).

        Update z_ij,k+1 according to equation (3b).
        """
        network.receive_all(agent)

        for j, msg in agent.messages.items():
            j_idx = agent.aux_vars["neighbor_to_idx"][j]
            new_z = (1 - self.alpha) * agent.aux_vars["z_i"][j_idx] - self.alpha * msg
            z_update = (
                (
                    self.ema_factor * agent.aux_vars["z_i"][j_idx]
                    + (1 - self.ema_factor) * new_z
                )
                if self.use_z_ema
                else new_z
            )
            agent.aux_vars["z_i"][j_idx] = z_update
