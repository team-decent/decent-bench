from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import decent_bench.utils.interoperability as iop
from decent_bench.agents import Agent
from decent_bench.algorithms.decentralized import P2PAlgorithm
from decent_bench.algorithms.utils import initial_states
from decent_bench.costs import PyTorchCost
from decent_bench.networks import P2PNetwork
from decent_bench.utils.types import InitialStates

if TYPE_CHECKING:
    import torch

try:
    import torch
except ImportError as e:
    raise ImportError(
        "PyTorch is required for LT-ADMM-EMA algorithm, but it is not installed. "
        "Please install PyTorch to use this algorithm."
    ) from e


@dataclass(eq=False)
class LT_ADMM_EMA(P2PAlgorithm):  # noqa: N801
    """
    Local Training ADMM with exponential moving averages algorithm for distributed optimization.

    Args:
        iterations: Total number of communication rounds (K)
        local_steps: Number of local training steps (tau)
        step_size: Local step size (gamma), can be a constant or a function of iteration
        aux_step_size: Local step size (beta), can be a constant or a function of iteration
        penalty: Penalty parameter (rho)
        alpha: Relaxation parameter (alpha)
        ema_factor: Exponential moving average factor, ema_factor * old + (1 - ema_factor) * new
        set_x_to_ema: Whether to set final x to EMA values in finalize()
        use_z_ema: Whether to use EMA for z_ij,k updates in auxiliary update
        send_ema_x: Whether to send EMA of x in communication phase instead of current x
        opt_cls: PyTorch optimizer class to use for local training. If provided, it will be used for local training
        opt_kwargs: Keyword arguments for PyTorch optimizer
        sched_cls: PyTorch scheduler class for local training (e.g., torch.optim.lr_scheduler.StepLR)
        sched_kwargs: Keyword arguments for PyTorch scheduler (e.g., {"step_size": 10, "gamma": 0.1})
        x0: Initial parameters (optional)
        name: Algorithm name (default "LT-ADMM-EMA")

    """

    iterations: int = 100  # Total number of communication rounds (K)
    local_steps: int = 5  # Number of local training steps (tau)
    step_size: float | Callable[[int], float] = 0.01  # Local step size (gamma)
    aux_step_size: float | Callable[[int], float] = 0.01  # Local step size (beta)
    penalty: float = 1.0  # Penalty parameter (rho)
    alpha: float = 0.5  # Relaxation parameter (alpha)
    ema_factor: float = 0.9  # Exponential moving average factor, ema_factor * old + (1 - ema_factor) * new
    set_x_to_ema: bool = True  # Whether to set final x to EMA values
    use_z_ema: bool = True  # Whether to use EMA for z_ij,k updates
    send_ema_x: bool = False  # Whether to send EMA of x in communication phase instead of current x
    opt_cls: type[torch.optim.Optimizer] | None = None  # PyTorch optimizer class to use for local training
    opt_kwargs: dict[str, Any] | None = None  # Keyword arguments for PyTorch optimizer
    sched_cls: type[torch.optim.lr_scheduler.LRScheduler] | None = None  # PyTorch scheduler class for local training
    sched_kwargs: dict[str, Any] | None = None  # Keyword arguments for PyTorch scheduler
    x0: InitialStates = None  # Initial parameters (optional)
    name: str = "LT-ADMM-EMA"

    def __post_init__(self) -> None:
        """
        Validate parameters.

        Raises:
            ValueError: If any of the parameters are invalid (e.g., non-positive iterations, local_steps, step_size,
            penalty, or alpha).

        """
        if self.local_steps <= 0:
            raise ValueError("local_steps must be positive")
        if isinstance(self.step_size, float) and self.step_size <= 0:
            raise ValueError("step_size must be positive")
        if callable(self.step_size):
            test_step_size = [self.step_size(k) for k in range(self.iterations)]
            if any(s <= 0 for s in test_step_size):
                raise ValueError("step_size function must return positive values for all iterations")
        if isinstance(self.aux_step_size, float) and self.aux_step_size <= 0:
            raise ValueError("aux_step_size must be positive")
        if callable(self.aux_step_size):
            test_aux_step_size = [self.aux_step_size(k) for k in range(self.iterations)]
            if any(s <= 0 for s in test_aux_step_size):
                raise ValueError("aux_step_size function must return positive values for all iterations")
        if self.penalty <= 0:
            raise ValueError("penalty must be positive")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")

    def initialize(self, network: P2PNetwork) -> None:  # noqa: D102
        self.x0 = initial_states(self.x0, network)

        # Initialize agents with auxiliary variables
        for i in network.agents():
            if not isinstance(i.cost, PyTorchCost):
                raise TypeError(f"LT-ADMM-EMA requires PyTorchCost, but agent {i} has cost of type {type(i.cost)}")

            # Initialize PyTorch optimizer for local training if use_torch_optim is True
            if self.opt_cls is not None:
                initial_step_size = self.step_size(0) if callable(self.step_size) else self.step_size
                if self.opt_kwargs is None:
                    self.opt_kwargs = {}
                self.opt_kwargs.setdefault("lr", initial_step_size)
                i.cost.init_local_training(
                    opt_cls=self.opt_cls,
                    opt_kwargs=self.opt_kwargs,
                    sched_cls=self.sched_cls,
                    sched_kwargs=self.sched_kwargs,
                )

            neighbors = network.neighbors(i)
            z_i = iop.zeros(
                shape=(len(neighbors), *iop.shape(self.x0[i])),
                framework=i.cost.framework,
                device=i.cost.device,
            )
            neighbor_to_idx = {}

            for idx, j in enumerate(neighbors):
                z_i[idx] = iop.copy(self.x0[i])
                neighbor_to_idx[j] = idx

            aux_vars = {
                "phi": self.x0[i],  # phi_i,k - model parameters
                "phi_ema": self.x0[i],  # Exponential moving average of phi_i,k
                "x_train": self.x0[i],  # x_i,k+1 - model parameters after local training
                "z_i": z_i,  # z_ij,k+1 - auxiliary consensus variable
                "neighbor_to_idx": neighbor_to_idx,
            }
            i.initialize(x=self.x0[i], aux_vars=aux_vars)

    def step(self, network: P2PNetwork, iteration: int) -> None:  # noqa: D102
        step_size = self.step_size(iteration) if callable(self.step_size) else self.step_size
        aux_step_size = self.aux_step_size(iteration) if callable(self.aux_step_size) else self.aux_step_size

        # Step 1: Local training phase
        for i in network.active_agents():
            self._local_training(i, network, step_size, aux_step_size)

        # Step 2: Communication phase
        for i in network.active_agents():
            self._communication(i, network)

        # Step 3: Auxiliary update phase
        for i in network.active_agents():
            self._auxiliary_update(i)

    def _local_training(self, agent: Agent, network: P2PNetwork, step_size: float, aux_step_size: float) -> None:
        """
        Perform local training steps.

        Updates phi_i,k and gradient estimators r_i,h,k.

        Raises:
            TypeError: If the agent's cost is not a PyTorchCost, since this implementation relies on
                PyTorch for local optimization.

        """
        if not isinstance(agent.cost, PyTorchCost):
            raise TypeError(f"LT-ADMM-EMA requires PyTorchCost, but agent {agent} has cost of type {type(agent.cost)}")

        neighbors = network.active_neighbors(agent)

        agent.aux_vars["phi"] = iop.copy(agent.aux_vars["x_train"])
        z_sum = iop.sum(agent.aux_vars["z_i"], dim=0)
        multiplier = self.penalty * len(neighbors)

        if self.opt_cls is not None:
            # Use PyTorch optimizer for local training
            for _ in range(self.local_steps):
                agent.aux_vars["phi"] = agent.cost.local_training(
                    x=agent.aux_vars["phi"],
                    iterations=1,
                    agent=agent,
                )
                agent.aux_vars["phi"] -= aux_step_size * (multiplier * agent.aux_vars["x_train"] - z_sum)
        else:
            for _ in range(self.local_steps):
                # Manual gradient step
                current_gradient = agent.cost.gradient(agent.aux_vars["phi"])
                step = step_size * current_gradient + aux_step_size * (multiplier * agent.aux_vars["x_train"] - z_sum)
                agent.aux_vars["phi"] -= step

        agent.aux_vars["phi_ema"] = (
            self.ema_factor * agent.aux_vars["phi_ema"] + (1 - self.ema_factor) * agent.aux_vars["phi"]
        )
        agent.aux_vars["x_train"] = agent.aux_vars["phi"]
        agent.x = agent.aux_vars["phi_ema"] if self.set_x_to_ema else agent.aux_vars["phi"]

    def _communication(self, agent: Agent, network: P2PNetwork) -> None:
        """
        Communication phase (Algorithm 1, line 11).

        Transmit z_ij,k - 2 * rho * x_i,k+1 to each neighbor.
        """
        for j in network.active_neighbors(agent):
            j_idx = agent.aux_vars["neighbor_to_idx"][j]
            message = agent.aux_vars["z_i"][j_idx] - 2 * self.penalty * (
                agent.x if self.send_ema_x else agent.aux_vars["x_train"]
            )
            network.send(agent, j, message)

    def _auxiliary_update(self, agent: Agent) -> None:
        """
        Auxiliary update phase (Algorithm 1, line 12).

        Update z_ij,k+1 according to equation (3b).
        """
        for j, msg in agent.messages.items():
            j_idx = agent.aux_vars["neighbor_to_idx"][j]
            new_z = (1 - self.alpha) * agent.aux_vars["z_i"][j_idx] - self.alpha * msg
            z_update = (
                (self.ema_factor * agent.aux_vars["z_i"][j_idx] + (1 - self.ema_factor) * new_z)
                if self.use_z_ema
                else new_z
            )
            agent.aux_vars["z_i"][j_idx] = z_update
