PyTorch Optimizer Integration
=============================

Following is a short example of how to use a PyTorch optimizer for local training in the LT-ADMM algorithm. 
This example defines a new algorithm class LT_ADMM_TORCH that inherits from LTADMM and overrides the local training step to use a PyTorch optimizer. 
The initialize method sets up the PyTorch optimizer for each agent, and the _local_training method performs local training using the optimizer.

.. code-block:: python

    from dataclasses import dataclass
    from typing import TYPE_CHECKING, Any

    import decent_bench.utils.interoperability as iop
    from decent_bench.agents import Agent
    from decent_bench.algorithms.p2p import LTADMM
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
    class LT_ADMM_TORCH(LTADMM):
        opt_cls: type[torch.optim.Optimizer] | None = None  # PyTorch optimizer class to use for local training
        opt_kwargs: dict[str, Any] | None = None  # Keyword arguments for PyTorch optimizer
        sched_cls: type[torch.optim.lr_scheduler.LRScheduler] | None = None  # PyTorch scheduler class for local training
        sched_kwargs: dict[str, Any] | None = None  # Keyword arguments for PyTorch scheduler
        name: str = "LT-ADMM-TORCH"

        def initialize(self, network: P2PNetwork) -> None:
            super().initialize(network)
            for i in network.agents():
                if not isinstance(i.cost, PyTorchCost):
                    raise TypeError(f"LT-ADMM-TORCH requires PyTorchCost, but agent {i} has cost of type {type(i.cost)}")

                # Initialize PyTorch optimizer for local training if use_torch_optim is True
                if self.opt_cls is not None:
                    if self.opt_kwargs is None:
                        self.opt_kwargs = {}
                    self.opt_kwargs.setdefault("lr", self.step_size)
                    i.cost.init_local_training(
                        opt_cls=self.opt_cls,
                        opt_kwargs=self.opt_kwargs,
                        sched_cls=self.sched_cls,
                        sched_kwargs=self.sched_kwargs,
                    )

        def _local_training(self, agent: Agent, network: P2PNetwork) -> None:
            if TYPE_CHECKING:
                if not isinstance(agent.cost, PyTorchCost):
                    raise TypeError(
                        f"LT-ADMM-TORCH requires PyTorchCost, but agent {agent} has cost of type {type(agent.cost)}"
                    )

            agent.aux_vars["phi"] = iop.copy(agent.x)
            z_sum = iop.sum(agent.aux_vars["z_i"], dim=0)
            multiplier = self.penalty * len(network.neighbors(agent))
            correction = self.aux_step_size * (multiplier * agent.x - z_sum)

            if self.opt_cls is not None:
                agent.aux_vars["phi"] = agent.cost.local_training(
                    x=agent.aux_vars["phi"],
                    iterations=self.num_local_steps,
                    regularization=correction,
                    agent=agent,
                )
            else:
                for _ in range(self.num_local_steps):
                    current_gradient = agent.cost.gradient(agent.aux_vars["phi"])
                    step = self.step_size * current_gradient + correction
                    # Update phi_i,k according to gradient step (line 7)
                    agent.aux_vars["phi"] -= step

            # Update agent's main parameter (line 10)
            agent.x = agent.aux_vars["phi"]
