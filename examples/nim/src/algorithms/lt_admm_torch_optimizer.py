from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import decent_bench.utils.interoperability as iop
from decent_bench.agents import Agent
from decent_bench.algorithms.decentralized import LT_ADMM
from decent_bench.costs import PyTorchCost
from decent_bench.networks import P2PNetwork

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
class LT_ADMM_TORCH(LT_ADMM):  # noqa: D101, N801
    opt_cls: type[torch.optim.Optimizer] | None = None  # PyTorch optimizer class to use for local training
    opt_kwargs: dict[str, Any] | None = None  # Keyword arguments for PyTorch optimizer
    sched_cls: type[torch.optim.lr_scheduler.LRScheduler] | None = None  # PyTorch scheduler class for local training
    sched_kwargs: dict[str, Any] | None = None  # Keyword arguments for PyTorch scheduler
    name: str = "LT-ADMM-TORCH"

    def initialize(self, network: P2PNetwork) -> None:  # noqa: D102
        super().initialize(network)
        for i in network.agents():
            if not isinstance(i.cost, PyTorchCost):
                raise TypeError(f"LT-ADMM-EMA requires PyTorchCost, but agent {i} has cost of type {type(i.cost)}")

            # Initialize PyTorch optimizer for local training if use_torch_optim is True
            if self.opt_cls is not None:
                initial_step_size = self.step_size(0) if callable(self.step_size) else self.step_size
                if self.opt_kwargs is None:
                    self.opt_kwargs = {}
                self.opt_kwargs["lr"] = initial_step_size
                i.cost.init_local_training(
                    opt_cls=self.opt_cls,
                    opt_kwargs=self.opt_kwargs,
                    sched_cls=self.sched_cls,
                    sched_kwargs=self.sched_kwargs,
                )

    def _local_training(self, agent: Agent, network: P2PNetwork, step_size: float, aux_step_size: float) -> None:
        if not isinstance(agent.cost, PyTorchCost):
            raise TypeError(f"LT-ADMM-EMA requires PyTorchCost, but agent {agent} has cost of type {type(agent.cost)}")

        neighbors = network.active_neighbors(agent)

        agent.aux_vars["phi"] = iop.copy(agent.x)
        z_sum = iop.sum(agent.aux_vars["z_i"], dim=0)
        multiplier = self.penalty * len(neighbors)

        for _ in range(self.local_steps):
            if self.opt_cls is not None:
                agent.aux_vars["phi"] = agent.cost.local_training(
                    x=agent.aux_vars["phi"],
                    iterations=1,
                    agent=agent,
                )
                agent.aux_vars["phi"] += aux_step_size * (multiplier * agent.x - z_sum)
            else:
                current_gradient = agent.cost.gradient(agent.aux_vars["phi"])
                step = step_size * current_gradient + aux_step_size * (multiplier * agent.x - z_sum)
                # Update phi_i,k according to gradient step (line 7)
                agent.aux_vars["phi"] -= step

        # Update agent's main parameter (line 10)
        agent.x = agent.aux_vars["phi"]
