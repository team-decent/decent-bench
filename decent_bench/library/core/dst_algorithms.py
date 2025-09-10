from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from decent_bench.library.core.network import Network


class DstAlgorithm(ABC):
    """Distributed algorithm - agents collaborate to solve an optimization problem using peer-to-peer communication."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the algorithm."""

    @abstractmethod
    def run(self, network: Network) -> None:
        """
        Run the algorithm.

        Args:
            network: provides agents, neighbors etc.

        """


@dataclass
class DGD(DstAlgorithm):
    r"""
    Distributed gradient descent characterized by the update step below.

    .. math::
        \mathbf{x}_{i, k+1} = (\sum_{j} \mathbf{w}_{ij} \mathbf{x}_{j,k}) - \rho \nabla f_i(\mathbf{x}_{i,k})

    where
    :math:`\mathbf{x}_{i, k}` is agent i's local optimization variable at iteration k,
    j is a neighbor of i or i itself,
    :math:`\mathbf{w}_{ij}` is the metropolis weight between agent i and j,
    :math:`\rho` is the step size,
    and :math:`f_i` is agent i's local cost function.

    """

    iterations: int
    step_size: float
    name: str = "DGD"  # pyright: ignore[reportIncompatibleMethodOverride]

    def run(self, network: Network) -> None:
        """
        Run the algorithm with all :math:`x_k` initialized using :func:`numpy.zeros`.

        Args:
            network: provides agents, neighbors etc.

        """
        for agent in network.get_all_agents():
            x_0 = np.zeros(agent.cost_function.domain_shape)
            agent.initialize(x=x_0, received_msgs=dict.fromkeys(network.get_neighbors(agent), x_0))
        W = network.metropolis_weights  # noqa: N806
        for k in range(self.iterations):
            for i in network.get_active_agents(k):
                neighborhood_avg = np.sum([W[i, j] * x_j for j, x_j in i.received_messages.items()], axis=0)
                neighborhood_avg += W[i, i] * i.x
                gradient = i.cost_function.gradient(i.x)
                i.x = neighborhood_avg - self.step_size * gradient
            for i in network.get_active_agents(k):
                network.broadcast(i, i.x)
            for i in network.get_active_agents(k):
                network.receive_all(i)


@dataclass
class GT1(DstAlgorithm):
    r"""
    Gradient tracking algorithm characterized by the update step below.

    .. math::
        \mathbf{y}_{i, k} = \mathbf{x}_{i, k} - \rho \nabla f_i(\mathbf{x}_{i,k})
    .. math::
        \mathbf{x}_{i, k+1} = \mathbf{y}_{i, k} - \mathbf{y}_{i, k-1} + \sum_j \mathbf{w}_{ij} \mathbf{x}_{j,k}

    where
    :math:`\mathbf{x}_{i, k}` is agent i's local optimization variable at iteration k,
    :math:`\rho` is the step size,
    :math:`f_i` is agent i's local cost function,
    j is a neighbor of i or i itself,
    and :math:`\mathbf{w}_{ij}` is the metropolis weight between agent i and j.

    """

    iterations: int
    step_size: float
    name: str = "GT1"  # pyright: ignore[reportIncompatibleMethodOverride]

    def run(self, network: Network) -> None:
        """
        Run the algorithm with all :math:`x_k` and :math:`y_{k-1}` initialized using :func:`numpy.zeros`.

        Args:
            network: provides agents, neighbors etc.

        """
        for agent in network.get_all_agents():
            x_0 = np.zeros(agent.cost_function.domain_shape)
            neighbors = network.get_neighbors(agent)
            agent.initialize(x=x_0, received_msgs=dict.fromkeys(neighbors, x_0), aux_vars={"y_old": x_0})
        W = network.metropolis_weights  # noqa: N806
        for k in range(self.iterations):
            for i in network.get_active_agents(k):
                i.aux_vars["y"] = i.x - self.step_size * i.cost_function.gradient(i.x)
                neighborhood_avg = np.sum([W[i, j] * x_j for j, x_j in i.received_messages.items()], axis=0)
                neighborhood_avg += W[i, i] * i.x
                i.x = i.aux_vars["y"] - i.aux_vars["y_old"] + neighborhood_avg
                i.aux_vars["y_old"] = i.aux_vars["y"]
            for i in network.get_active_agents(k):
                network.broadcast(i, i.x)
            for i in network.get_active_agents(k):
                network.receive_all(i)
