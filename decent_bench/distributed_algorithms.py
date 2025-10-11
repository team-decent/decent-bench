from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from decent_bench.network import Network


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


@dataclass(eq=False)
class DGD(DstAlgorithm):
    r"""
    Distributed gradient descent characterized by the update step below.

    .. math::
        \mathbf{x}_{i, k+1} = (\sum_{j} \mathbf{W}_{ij} \mathbf{x}_{j,k}) - \rho \nabla f_i(\mathbf{x}_{i,k})

    where
    :math:`\mathbf{x}_{i, k}` is agent i's local optimization variable at iteration k,
    j is a neighbor of i or i itself,
    :math:`\mathbf{W}_{ij}` is the metropolis weight between agent i and j,
    :math:`\rho` is the step size,
    and :math:`f_i` is agent i's local cost function.

    """

    iterations: int
    step_size: float
    name: str = "DGD"

    def run(self, network: Network) -> None:
        r"""
        Run the algorithm with all :math:`\mathbf{x}` initialized using :func:`numpy.zeros`.

        Args:
            network: provides agents, neighbors etc.

        """
        for agent in network.get_all_agents():
            x0 = np.zeros(agent.cost_function.domain_shape)
            agent.initialize(x=x0, received_msgs=dict.fromkeys(network.get_neighbors(agent), x0))
        W = network.metropolis_weights  # noqa: N806
        for k in range(self.iterations):
            for i in network.get_active_agents(k):
                neighborhood_avg = np.sum([W[i, j] * x_j for j, x_j in i.received_messages.items()], axis=0)
                neighborhood_avg += W[i, i] * i.x
                i.x = neighborhood_avg - self.step_size * i.cost_function.gradient(i.x)
            for i in network.get_active_agents(k):
                network.broadcast(i, i.x)
            for i in network.get_active_agents(k):
                network.receive_all(i)

@dataclass(eq=False)
class ATC(DstAlgorithm):
    r"""
    Distributed gradient descent characterized by the update step below, called Adapt-Then-Combine (ATC) [1]. Alias `AdaptThenCombine`.

    .. math::
        \mathbf{x}_{i, k+1} = (\sum_{j} \mathbf{W}_{ij} \mathbf{x}_{j,k} - \rho \nabla f_j(\mathbf{x}_{j,k}))

    where
    :math:`\mathbf{x}_{i, k}` is agent i's local optimization variable at iteration k,
    j is a neighbor of i or i itself,
    :math:`\mathbf{W}_{ij}` is the metropolis weight between agent i and j,
    :math:`\rho` is the step size,
    and :math:`f_i` is agent i's local cost function.

    .. [1] J. Chen and A. H. Sayed, "Diffusion Adaptation Strategies for Distributed Optimization and Learning Over Networks," IEEE Trans. Signal Process., vol. 60, no. 8, pp. 4289-4305, Aug. 2012, doi: 10.1109/TSP.2012.2198470.

    """

    iterations: int
    step_size: float
    name: str = "ATC"

    def run(self, network: Network) -> None:
        r"""
        Run the algorithm with all :math:`\mathbf{x}` initialized using :func:`numpy.zeros`.

        Args:
            network: provides agents, neighbors etc.

        """
        for agent in network.get_all_agents():
            x0 = np.zeros(agent.cost_function.domain_shape)
            agent.initialize(x=x0, received_msgs=dict.fromkeys(network.get_neighbors(agent), x0), aux_vars={"y": x0})
        W = network.metropolis_weights  # noqa: N806

        for k in range(self.iterations):
            # gradient step (adapt)
            for i in network.get_active_agents(k):
                i.aux_vars["y"] = i.x - self.step_size * i.cost_function.gradient(i.x)
            # transmit and receive
            for i in network.get_active_agents(k):
                network.broadcast(i, i.aux_vars["y"])
            # consensus (combine)
            for i in network.get_active_agents(k):
                network.receive_all(i)
                neighborhood_avg = np.sum([W[i, j] * x_j for j, x_j in i.received_messages.items()], axis=0)
                neighborhood_avg += W[i, i] * i.x
                i.x = neighborhood_avg

AdaptThenCombine = ATC # alias

@dataclass(eq=False)
class GT1(DstAlgorithm):
    r"""
    Gradient tracking algorithm characterized by the update step below.

    .. math::
        \mathbf{y}_{i, k+1} = \mathbf{x}_{i, k} - \rho \nabla f_i(\mathbf{x}_{i,k})
    .. math::
        \mathbf{x}_{i, k+1} = \mathbf{y}_{i, k+1} - \mathbf{y}_{i, k} + \sum_j \mathbf{W}_{ij} \mathbf{x}_{j,k}

    where
    :math:`\mathbf{x}_{i, k}` is agent i's local optimization variable at iteration k,
    :math:`\rho` is the step size,
    :math:`f_i` is agent i's local cost function,
    j is a neighbor of i or i itself,
    and :math:`\mathbf{W}_{ij}` is the metropolis weight between agent i and j.

    """

    iterations: int
    step_size: float
    name: str = "GT1"

    def run(self, network: Network) -> None:
        r"""
        Run the algorithm with all :math:`\mathbf{x}` and :math:`\mathbf{y}` initialized using :func:`numpy.zeros`.

        Args:
            network: provides agents, neighbors etc.

        """
        for agent in network.get_all_agents():
            x0 = np.zeros(agent.cost_function.domain_shape)
            y0 = np.zeros(agent.cost_function.domain_shape)
            neighbors = network.get_neighbors(agent)
            agent.initialize(x=x0, received_msgs=dict.fromkeys(neighbors, x0), aux_vars={"y": y0})
        W = network.metropolis_weights  # noqa: N806
        for k in range(self.iterations):
            for i in network.get_active_agents(k):
                i.aux_vars["y_new"] = i.x - self.step_size * i.cost_function.gradient(i.x)
                neighborhood_avg = np.sum([W[i, j] * x_j for j, x_j in i.received_messages.items()], axis=0)
                neighborhood_avg += W[i, i] * i.x
                i.x = i.aux_vars["y_new"] - i.aux_vars["y"] + neighborhood_avg
                i.aux_vars["y"] = i.aux_vars["y_new"]
            for i in network.get_active_agents(k):
                network.broadcast(i, i.x)
            for i in network.get_active_agents(k):
                network.receive_all(i)


@dataclass(eq=False)
class GT2(DstAlgorithm):
    r"""
    Gradient tracking algorithm characterized by the update step below.

    .. math::
        \mathbf{y}_{i, k+1} = \mathbf{x}_{i, k} - \rho \nabla f_i(\mathbf{x}_{i,k})
    .. math::
        \mathbf{x}_{i, k+1}
        = \sum_j \frac{1}{2} (\mathbf{I} + \mathbf{W})_{ij} (\mathbf{x}_{j,k} + \mathbf{y}_{j, k+1} - \mathbf{y}_{j, k})

    where
    :math:`\mathbf{x}_{i, k}` is agent i's local optimization variable at iteration k,
    :math:`\rho` is the step size,
    :math:`f_i` is agent i's local cost function,
    j is a neighbor of i or i itself,
    and :math:`\mathbf{W}_{ij}` is the metropolis weight between agent i and j.

    """

    iterations: int
    step_size: float
    name: str = "GT2"

    def run(self, network: Network) -> None:
        r"""
        Run the algorithm with all :math:`\mathbf{x}` and :math:`\mathbf{y}` initialized using :func:`numpy.zeros`.

        Args:
            network: provides agents, neighbors etc.

        """
        for i in network.get_all_agents():
            x0 = np.zeros(i.cost_function.domain_shape)
            y0 = np.zeros(i.cost_function.domain_shape)
            y1 = x0 - self.step_size * i.cost_function.gradient(x0)
            # note: msg0's y1 is an approximation of the neighbors' y1 (x0 and y0 are exact: all agents start with same)
            msg0 = x0 + y1 - y0
            i.initialize(
                x=x0,
                aux_vars={"y": y0, "y_new": y1},
                received_msgs=dict.fromkeys(network.get_neighbors(i), msg0),
            )
        W = 0.5 * (np.eye(*(network.metropolis_weights.shape)) + network.metropolis_weights)  # noqa: N806
        for k in range(self.iterations):
            for i in network.get_active_agents(k):
                i.x = np.sum([W[i, j] * msg for j, msg in i.received_messages.items()], axis=0) + W[i, i] * (
                    i.x + i.aux_vars["y_new"] - i.aux_vars["y"]
                )
                i.aux_vars["y"] = i.aux_vars["y_new"]
                i.aux_vars["y_new"] = i.x - self.step_size * i.cost_function.gradient(i.x)
            for i in network.get_active_agents(k):
                network.broadcast(i, i.x + i.aux_vars["y_new"] - i.aux_vars["y"])
            for i in network.get_active_agents(k):
                network.receive_all(i)

@dataclass(eq=False)
class AugDGM(DstAlgorithm):
    r"""
    Gradient tracking algorithm characterized by the update step below, called Aug-DGM [1] or ATC-DIGing [2]. Alias `ATCDIGing`.

    .. math::
        \mathbf{x}_{i, k+1} = \sum_j \mathbf{W}_{ij} (\mathbf{x}_{j, k} - \rho \mathbf{y}_{j, k})
    .. math::
        \mathbf{y}_{i, k+1} = \sum_j \mathbf{W}_{ij} (\mathbf{y}_{j, k} + \nabla f_j(\mathbf{x}_{j,k+1}) - \nabla f_j(\mathbf{x}_{j,k}))

    where
    :math:`\mathbf{x}_{i, k}` is agent i's local optimization variable at iteration k,
    :math:`\rho` is the step size,
    :math:`f_i` is agent i's local cost function,
    j is a neighbor of i or i itself,
    and :math:`\mathbf{W}_{ij}` is the metropolis weight between agent i and j.

    .. [1] J. Xu, S. Zhu, Y. C. Soh, and L. Xie, "Augmented distributed gradient methods for multi-agent optimization under uncoordinated constant stepsizes," in 2015 54th IEEE Conference on Decision and Control (CDC), Osaka, Japan: IEEE, Dec. 2015, pp. 2055-2060. doi: 10.1109/CDC.2015.7402509.
    .. [2] A. Nedic, A. Olshevsky, W. Shi, and C. A. Uribe, "Geometrically convergent distributed optimization with uncoordinated step-sizes," in 2017 American Control Conference (ACC), Seattle, WA, USA: IEEE, May 2017, pp. 3950–3955. doi: 10.23919/ACC.2017.7963560.

    """

    iterations: int
    step_size: float
    name: str = "Aug-DGM"

    def run(self, network: Network) -> None:
        r"""
        Run the algorithm with all :math:`\mathbf{x}` and :math:`\mathbf{y}` initialized using :func:`numpy.zeros`.

        Args:
            network: provides agents, neighbors etc.

        """
        for i in network.get_all_agents():
            x0 = np.zeros(i.cost_function.domain_shape)
            y0 = i.cost_function.gradient(x0)
            neighbors = network.get_neighbors(i)
            i.initialize(x=x0, received_msgs=dict.fromkeys(neighbors, x0), aux_vars={"y": y0, "g": y0, "g_new": x0, "s": x0})

        W = network.metropolis_weights  # noqa: N806

        for k in range(self.iterations):
            # 1st communication round
            #     step 1: perform local gradient step and communicate
            for i in network.get_active_agents(k):
                i.aux_vars["s"] = i.x - self.step_size * i.aux_vars["y"]
                network.broadcast(i, i.aux_vars["s"])
            
            #     step 2: update state and compute new local gradient
            for i in network.get_active_agents(k):
                network.receive_all(i)
                neighborhood_avg = np.sum([W[i, j] * s_j for j, s_j in i.received_messages.items()], axis=0)
                neighborhood_avg += W[i, i] * i.aux_vars["s"]
                i.x = neighborhood_avg    
                i.aux_vars["g_new"] = i.cost_function.gradient(i.x)
            
            # 2nd communication round
            #     step 1: transmit local gradient tracker
            for i in network.get_active_agents(k):
                network.broadcast(i, i.aux_vars["y"] + i.aux_vars["g_new"] - i.aux_vars["g"])
            
            #     step 2: update y (global gradient estimator)
            for i in network.get_active_agents(k):
                network.receive_all(i)
                neighborhood_avg = np.sum([W[i, j] * q_j for j, q_j in i.received_messages.items()], axis=0)
                neighborhood_avg += W[i, i] * (i.aux_vars["y"] + i.aux_vars["g_new"] - i.aux_vars["g"])
                i.aux_vars["y"] = neighborhood_avg
                i.aux_vars["g"] = i.aux_vars["g_new"]

ATCDIGing = AugDGM # alias

@dataclass(eq=False)
class ADMM(DstAlgorithm):
    r"""
    Distributed Alternating Direction Method of Multipliers characterized by the update step below.

    .. math::
        \mathbf{x}_{i, k+1} = \operatorname{prox}_{\frac{1}{\rho N_i} f_i}
        \left(\sum_j \mathbf{Z}_{ij, k} \frac{1}{\rho N_i} \right)
    .. math::
        \mathbf{Z}_{ij, k+1} = (1-\alpha) \mathbf{Z}_{ij, k} - \alpha (\mathbf{Z}_{ji, k} - 2 \rho \mathbf{x}_{j, k+1})

    where
    :math:`\mathbf{x}_{i, k}` is agent i's local optimization variable at iteration k,
    :math:`\operatorname{prox}` is the proximal operator described in :meth:`CostFunction.proximal()
    <decent_bench.cost_functions.CostFunction.proximal>`,
    :math:`\rho > 0` is the Lagrangian penalty parameter,
    :math:`N_i` is the number of neighbors of i,
    :math:`f_i` is i's local cost function,
    j is a neighbor of i,
    and :math:`\alpha \in (0, 1)` is the relaxation parameter.

    """

    iterations: int
    rho: float
    alpha: float
    name: str = "ADMM"

    def run(self, network: Network) -> None:
        r"""
        Run the algorithm with :math:`\mathbf{Z}` initialized using :func:`numpy.zeros`.

        Args:
            network: provides agents, neighbors etc.

        """
        pN = {i: self.rho * len(network.get_neighbors(i)) for i in network.get_all_agents()}  # noqa: N806
        all_agents = network.get_all_agents()
        for agent in all_agents:
            z0 = np.zeros((len(all_agents), *(agent.cost_function.domain_shape)))
            x1 = agent.cost_function.proximal(y=np.sum(z0, axis=0) / pN[agent], rho=1 / pN[agent])
            # note: msg0's x1 is an approximation of the neighbors' x1 (z0 is exact: all agents start with same)
            msg0: NDArray[float64] = z0[agent] - 2 * self.rho * x1
            agent.initialize(
                x=x1,
                aux_vars={"z": z0},
                received_msgs=dict.fromkeys(network.get_neighbors(agent), msg0),
            )
        for k in range(self.iterations):
            for i in network.get_active_agents(k):
                i.x = i.cost_function.proximal(y=np.sum(i.aux_vars["z"], axis=0) / pN[i], rho=1 / pN[i])
            for i in network.get_active_agents(k):
                for j in network.get_neighbors(i):
                    network.send(i, j, i.aux_vars["z"][j] - 2 * self.rho * i.x)
            for i in network.get_active_agents(k):
                network.receive_all(i)
            for i in network.get_active_agents(k):
                for j in network.get_neighbors(i):
                    i.aux_vars["z"][j] = (1 - self.alpha) * i.aux_vars["z"][j] - self.alpha * (i.received_messages[j])
