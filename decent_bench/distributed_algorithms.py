from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy import float64
from numpy.typing import NDArray

import decent_bench.utils.interoperability as iop
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
        Run the algorithm.

        Args:
            network: provides agents, neighbors etc.

        """
        for agent in network.get_all_agents():
            x0 = np.zeros(agent.cost_function.domain_shape)
            agent.initialize(x=x0, received_msgs=dict.fromkeys(network.get_neighbors(agent), x0))
        W = iop.from_numpy_like(network.metropolis_weights, network.get_all_agents()[0].x)  # noqa: N806
        for k in range(self.iterations):
            for i in network.get_active_agents(k):
                neighborhood_avg = iop.sum([W[i, j] * x_j for j, x_j in i.received_messages.items()], dim=0)
                neighborhood_avg += W[i, i] * i.x
                i.x = neighborhood_avg - self.step_size * i.cost_function.gradient(i.x)
            for i in network.get_active_agents(k):
                network.broadcast(i, i.x)
            for i in network.get_active_agents(k):
                network.receive_all(i)


@dataclass(eq=False)
class ATC(DstAlgorithm):
    r"""
    Adapt-Then-Combine (ATC) distributed gradient descent characterized by the update below [r1]_.

    .. math::
        \mathbf{x}_{i, k+1} = (\sum_{j} \mathbf{W}_{ij} \mathbf{x}_{j,k} - \rho \nabla f_j(\mathbf{x}_{j,k}))

    where
    :math:`\mathbf{x}_{i, k}` is agent i's local optimization variable at iteration k,
    j is a neighbor of i or i itself,
    :math:`\mathbf{W}_{ij}` is the metropolis weight between agent i and j,
    :math:`\rho` is the step size,
    and :math:`f_i` is agent i's local cost function.

    Alias: :class:`AdaptThenCombine`

    .. [r1] J. Chen and A. H. Sayed, "Diffusion Adaptation Strategies for Distributed Optimization and
            Learning Over Networks," IEEE Trans. Signal Process., vol. 60, no. 8, pp. 4289-4305,
            Aug. 2012, doi: 10.1109/TSP.2012.2198470.

    """

    iterations: int
    step_size: float
    name: str = "ATC"

    def run(self, network: Network) -> None:
        r"""
        Run the algorithm.

        Args:
            network: provides agents, neighbors etc.

        """
        for agent in network.get_all_agents():
            x0 = np.zeros(agent.cost_function.domain_shape)
            agent.initialize(
                x=x0,
                received_msgs=dict.fromkeys(network.get_neighbors(agent), x0),
                aux_vars={"y": x0},
            )
        W = iop.from_numpy_like(network.metropolis_weights, network.get_all_agents()[0].x)  # noqa: N806

        for k in range(self.iterations):
            # gradient step (a.k.a. adapt step)
            for i in network.get_active_agents(k):
                i.aux_vars["y"] = i.x - self.step_size * i.cost_function.gradient(i.x)

            # transmit and receive
            for i in network.get_active_agents(k):
                network.broadcast(i, i.aux_vars["y"])

            # consensus (a.k.a. combine step)
            for i in network.get_active_agents(k):
                network.receive_all(i)

            for i in network.get_active_agents(k):
                s = iop.stack([W[i, j] * x_j for j, x_j in i.received_messages.items()], dim=0)
                neighborhood_avg = iop.sum(s, dim=0)
                neighborhood_avg += W[i, i] * i.x
                i.x = neighborhood_avg


AdaptThenCombine = ATC  # alias


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
        Run the algorithm.

        Args:
            network: provides agents, neighbors etc.

        """
        for agent in network.get_all_agents():
            x0 = np.zeros(agent.cost_function.domain_shape)
            y0 = np.zeros(agent.cost_function.domain_shape)
            neighbors = network.get_neighbors(agent)
            agent.initialize(x=x0, received_msgs=dict.fromkeys(neighbors, x0), aux_vars={"y": y0})
        W = iop.from_numpy_like(network.metropolis_weights, network.get_all_agents()[0].x)  # noqa: N806
        for k in range(self.iterations):
            for i in network.get_active_agents(k):
                i.aux_vars["y_new"] = i.x - self.step_size * i.cost_function.gradient(i.x)
                neighborhood_avg = iop.sum([W[i, j] * x_j for j, x_j in i.received_messages.items()], dim=0)
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
        Run the algorithm.

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
        W = iop.from_numpy_like(  # noqa: N806
            0.5 * (np.eye(*(network.metropolis_weights.shape)) + network.metropolis_weights),
            network.get_all_agents()[0].x,
        )
        for k in range(self.iterations):
            for i in network.get_active_agents(k):
                i.x = iop.sum([W[i, j] * msg for j, msg in i.received_messages.items()], dim=0) + W[i, i] * (
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
    Aug-DGM [r2]_ or ATC-DIGing [r3]_ gradient tracking algorithm, characterized by the updates below.

    .. math::
        \mathbf{x}_{i, k+1} = \sum_j \mathbf{W}_{ij} (\mathbf{x}_{j, k} - \rho \mathbf{y}_{j, k})
    .. math::
        \mathbf{y}_{i, k+1} = \sum_j \mathbf{W}_{ij} (\mathbf{y}_{j, k}
                            + \nabla f_j(\mathbf{x}_{j,k+1}) - \nabla f_j(\mathbf{x}_{j,k}))

    where
    :math:`\mathbf{x}_{i, k}` is agent i's local optimization variable at iteration k,
    :math:`\rho` is the step size,
    :math:`f_i` is agent i's local cost function,
    j is a neighbor of i or i itself,
    and :math:`\mathbf{W}_{ij}` is the metropolis weight between agent i and j.

    Alias: :class:`ATCDIGing`

    .. [r2] J. Xu, S. Zhu, Y. C. Soh, and L. Xie, "Augmented distributed gradient methods for multi-agent
            optimization under uncoordinated constant stepsizes," in 2015 54th IEEE Conference on Decision
            and Control (CDC), Osaka, Japan: IEEE, Dec. 2015, pp. 2055-2060. doi: 10.1109/CDC.2015.7402509.
    .. [r3] A. Nedic, A. Olshevsky, W. Shi, and C. A. Uribe, "Geometrically convergent distributed
            optimization with uncoordinated step-sizes," in 2017 American Control Conference (ACC), Seattle,
            WA, USA: IEEE, May 2017, pp. 3950-3955. doi: 10.23919/ACC.2017.7963560.

    """

    iterations: int
    step_size: float
    name: str = "Aug-DGM"

    def run(self, network: Network) -> None:
        r"""
        Run the algorithm.

        Args:
            network: provides agents, neighbors etc.

        """
        for i in network.get_all_agents():
            x0 = np.zeros(i.cost_function.domain_shape)
            y0 = i.cost_function.gradient(x0)
            neighbors = network.get_neighbors(i)
            i.initialize(
                x=x0,
                received_msgs=dict.fromkeys(neighbors, x0),
                aux_vars={"y": y0, "g": y0, "g_new": x0, "s": x0},
            )

        W = iop.from_numpy_like(network.metropolis_weights, network.get_all_agents()[0].x)  # noqa: N806

        for k in range(self.iterations):
            # 1st communication round
            #     step 1: perform local gradient step and communicate
            for i in network.get_active_agents(k):
                i.aux_vars["s"] = i.x - self.step_size * i.aux_vars["y"]

            for i in network.get_active_agents(k):
                network.broadcast(i, i.aux_vars["s"])

            #     step 2: update state and compute new local gradient
            for i in network.get_active_agents(k):
                network.receive_all(i)

            for i in network.get_active_agents(k):
                s = iop.stack([W[i, j] * s_j for j, s_j in i.received_messages.items()], dim=0)
                neighborhood_avg = iop.sum(s, dim=0)
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

            for i in network.get_active_agents(k):
                s = iop.stack([W[i, j] * q_j for j, q_j in i.received_messages.items()], dim=0)
                neighborhood_avg = iop.sum(s, dim=0)
                neighborhood_avg += W[i, i] * (i.aux_vars["y"] + i.aux_vars["g_new"] - i.aux_vars["g"])
                i.aux_vars["y"] = neighborhood_avg
                i.aux_vars["g"] = i.aux_vars["g_new"]


ATCDIGing = AugDGM  # alias


@dataclass(eq=False)
class WangElia(DstAlgorithm):
    r"""
    Wang-Elia gradient tracking algorithm characterized by the updates below, see [r4]_ and [r5]_.

    .. math::
        \mathbf{x}_{i, k+1} = \mathbf{x}_{i, k} - \sum_j \mathbf{K}_{ij} (\mathbf{x}_{j, k} + \mathbf{z}_{j, k})
                            - \rho \nabla f_i(\mathbf{x}_{i,k})
    .. math::
        \mathbf{z}_{i, k+1} = \mathbf{z}_{i, k} + \sum_j \mathbf{K}_{ij} \mathbf{x}_{j, k}

    where
    :math:`\mathbf{x}_{i, k}` is agent i's local optimization variable at iteration k,
    :math:`\rho` is the step size,
    :math:`f_i` is agent i's local cost function,
    j is a neighbor of i or i itself,
    and :math:`\mathbf{K}_{ij}` is the weight between agent i and j.
    The matrix :math:`\mathbf{K}` is chosen as :math:`0.5 (\mathbf{I} - \mathbf{W})`,
    where :math:`\mathbf{W}` is the Metropolis weight matrix.

    .. [r4] J. Wang and N. Elia, "Control approach to distributed optimization," in 2010 48th Annual Allerton
            Conference on Communication, Control, and Computing (Allerton), Monticello, IL, USA: IEEE, Sep. 2010,
            pp. 557-561. doi: 10.1109/ALLERTON.2010.5706956.
    .. [r5] M. Bin, I. Notarnicola, and T. Parisini, "Stability, Linear Convergence, and Robustness of the
            Wang-Elia Algorithm for Distributed Consensus Optimization," in 2022 IEEE 61st Conference on
            Decision and Control (CDC), Cancun, Mexico: IEEE, Dec. 2022, pp. 1610-1615.
            doi: 10.1109/CDC51059.2022.9993284.

    """

    iterations: int
    step_size: float
    name: str = "Wang-Elia"

    def run(self, network: Network) -> None:
        r"""
        Run the algorithm.

        Args:
            network: provides agents, neighbors etc.

        """
        for i in network.get_all_agents():
            x0 = np.zeros(i.cost_function.domain_shape)
            neighbors = network.get_neighbors(i)
            i.initialize(
                x=x0,
                received_msgs=dict.fromkeys(neighbors, x0),
                aux_vars={"z": x0, "x_old": x0},
            )

        W = iop.from_numpy_like(network.metropolis_weights, network.get_all_agents()[0].x)  # noqa: N806
        K = 0.5 * (iop.eye_like(W) - W)  # noqa: N806

        for k in range(self.iterations):
            # 1st communication round
            for i in network.get_active_agents(k):
                network.broadcast(i, i.x + i.aux_vars["z"])

            # do consensus and local gradient step
            for i in network.get_active_agents(k):
                network.receive_all(i)

            for i in network.get_active_agents(k):
                neighborhood_avg = iop.sum([K[i, j] * m_j for j, m_j in i.received_messages.items()], dim=0)
                neighborhood_avg += K[i, i] * (i.x + i.aux_vars["z"])

                i.aux_vars["x_old"] = i.x
                i.x = i.x - neighborhood_avg - self.step_size * i.cost_function.gradient(i.x)

            # 2nd communication round
            for i in network.get_active_agents(k):
                network.broadcast(i, i.aux_vars["x_old"])

            # update auxiliary variable
            for i in network.get_active_agents(k):
                network.receive_all(i)

            for i in network.get_active_agents(k):
                neighborhood_avg = iop.sum([K[i, j] * m_j for j, m_j in i.received_messages.items()], dim=0)
                neighborhood_avg += K[i, i] * i.aux_vars["x_old"]
                i.aux_vars["z"] += neighborhood_avg


@dataclass(eq=False)
class EXTRA(DstAlgorithm):
    r"""
    EXTRA [r6]_ gradient tracking algorithm characterized by the update steps below.

    .. math::
        \mathbf{x}_{i, k+1}
        = \mathbf{x}_{i, k} + \sum_j \mathbf{W}_{ij} \mathbf{x}_{j,k}
        - \sum_j \tilde{\mathbf{W}}_{ij} \mathbf{x}_{j,k-1}
        - \rho (\nabla f_i(\mathbf{x}_{i,k}) - \nabla f_i(\mathbf{x}_{i,k-1}))

    where
    :math:`\mathbf{x}_{i, k}` is agent i's local optimization variable at iteration k,
    :math:`\rho` is the step size,
    :math:`f_i` is agent i's local cost function,
    j is a neighbor of i or i itself,
    :math:`\mathbf{W}_{ij}` is the metropolis weight between agent i and j,
    and :math:`\tilde{\mathbf{W}} = (\mathbf{I} + \mathbf{W}) / 2`.

    .. [r6] W. Shi, Q. Ling, G. Wu, and W. Yin, "EXTRA: An Exact First-Order Algorithm for Decentralized
            Consensus Optimization," SIAM J. Optim., vol. 25, no. 2, pp. 944-966, Jan. 2015,
            doi: 10.1137/14096668X.

    """

    iterations: int
    step_size: float
    name: str = "EXTRA"

    def run(self, network: Network) -> None:
        r"""
        Run the algorithm.

        Args:
            network: provides agents, neighbors etc.

        """
        # initialization (iteration k=0)
        for i in network.get_all_agents():
            x0 = np.zeros(i.cost_function.domain_shape)
            i.initialize(
                x=x0,
                received_msgs=dict.fromkeys(network.get_neighbors(i), x0),
                aux_vars={"x_old": x0, "x_old_old": x0, "x_cons": x0},
            )

        W = iop.from_numpy_like(network.metropolis_weights, network.get_all_agents()[0].x)  # noqa: N806

        # first iteration (iteration k=1)
        for i in network.get_active_agents(0):
            network.broadcast(i, i.x)
        for i in network.get_active_agents(0):
            network.receive_all(i)
        for i in network.get_active_agents(0):
            s = iop.stack([W[i, j] * x_j for j, x_j in i.received_messages.items()], dim=0)
            neighborhood_avg = iop.sum(s, dim=0)
            neighborhood_avg += W[i, i] * i.x
            i.aux_vars["x_cons"] = neighborhood_avg  # store W x_k
            i.aux_vars["x_old"] = i.x  # store x_0
            i.x = neighborhood_avg - self.step_size * i.cost_function.gradient(i.x)

        # subsequent iterations (k >= 2)
        for k in range(1, self.iterations):
            for i in network.get_active_agents(k):
                network.broadcast(i, i.x)
            for i in network.get_active_agents(k):
                network.receive_all(i)
            for i in network.get_active_agents(k):
                s = iop.stack([W[i, j] * x_j for j, x_j in i.received_messages.items()], dim=0)
                neighborhood_avg = iop.sum(s, dim=0)
                neighborhood_avg += W[i, i] * i.x
                i.aux_vars["x_old_old"] = i.aux_vars["x_old"]  # store x_{k-1}
                i.aux_vars["x_old"] = i.x  # store x_k
                # update x_{k+1}
                i.x = (
                    i.x
                    + neighborhood_avg
                    - 0.5 * i.aux_vars["x_old_old"]
                    - 0.5 * i.aux_vars["x_cons"]
                    - self.step_size
                    * (i.cost_function.gradient(i.x) - i.cost_function.gradient(i.aux_vars["x_old_old"]))
                )
                i.aux_vars["x_cons"] = neighborhood_avg  # store W x_k


@dataclass(eq=False)
class ATCTracking(DstAlgorithm):
    r"""
    ATC-Tracking [r7]_, [r8]_, [r9]_ gradient tracking algorithm, characterized by the updates below.

    .. math::
        \mathbf{x}_{i, k+1} = \sum_j \mathbf{W}_{ij} (\mathbf{x}_{j, k} - \rho \mathbf{y}_{j, k})
    .. math::
        \mathbf{y}_{i, k+1} = \sum_j \mathbf{W}_{ij} \mathbf{y}_{j, k}
                            + \nabla f_i(\mathbf{x}_{i,k+1}) - \nabla f_i(\mathbf{x}_{i,k})

    where
    :math:`\mathbf{x}_{i, k}` is agent i's local optimization variable at iteration k,
    :math:`\rho` is the step size,
    :math:`f_i` is agent i's local cost function,
    j is a neighbor of i or i itself,
    and :math:`\mathbf{W}_{ij}` is the metropolis weight between agent i and j.

    Aliases: :class:`SONATA`, :class:`NEXT`, :class:`ATCT`

    .. [r7] P. Di Lorenzo and G. Scutari, "NEXT: In-Network Nonconvex Optimization," IEEE Transactions
            on Signal and Information Processing over Networks, vol. 2, no. 2, pp. 120-136, Jun. 2016,
            doi: 10.1109/TSIPN.2016.2524588.

    .. [r8] G. Scutari and Y. Sun, "Distributed nonconvex constrained optimization over time-varying
            digraphs," Math. Program., vol. 176, no. 1-2, pp. 497-544, Jul. 2019, doi: 10.1007/s10107-018-01357-w.

    .. [r9] S. A. Alghunaim, E. K. Ryu, K. Yuan, and A. H. Sayed, "Decentralized Proximal Gradient Algorithms
            With Linear Convergence Rates," IEEE Transactions on Automatic Control, vol. 66, no. 6, pp. 2787-2794,
            Jun. 2021, doi: 10.1109/TAC.2020.3009363.

    """

    iterations: int
    step_size: float
    name: str = "ATC-Tracking"

    def run(self, network: Network) -> None:
        r"""
        Run the algorithm.

        Args:
            network: provides agents, neighbors etc.

        """
        for i in network.get_all_agents():
            x0 = np.zeros(i.cost_function.domain_shape)
            y0 = i.cost_function.gradient(x0)
            neighbors = network.get_neighbors(i)
            i.initialize(
                x=x0,
                received_msgs=dict.fromkeys(neighbors, x0),
                aux_vars={"y": y0, "g": y0, "g_new": x0, "s": x0},
            )

        W = iop.from_numpy_like(network.metropolis_weights, network.get_all_agents()[0].x)  # noqa: N806

        for k in range(self.iterations):
            # 1st communication round
            #     step 1: perform local gradient step and communicate
            for i in network.get_active_agents(k):
                i.aux_vars["s"] = i.x - self.step_size * i.aux_vars["y"]

            for i in network.get_active_agents(k):
                network.broadcast(i, i.aux_vars["s"])

            #     step 2: update state and compute new local gradient
            for i in network.get_active_agents(k):
                network.receive_all(i)

            for i in network.get_active_agents(k):
                s = iop.stack([W[i, j] * s_j for j, s_j in i.received_messages.items()], dim=0)
                neighborhood_avg = iop.sum(s, dim=0)
                neighborhood_avg += W[i, i] * i.aux_vars["s"]
                i.x = neighborhood_avg
                i.aux_vars["g_new"] = i.cost_function.gradient(i.x)

            # 2nd communication round
            #     step 1: transmit local gradient tracker
            for i in network.get_active_agents(k):
                network.broadcast(i, i.aux_vars["y"])

            #     step 2: update y (global gradient estimator)
            for i in network.get_active_agents(k):
                network.receive_all(i)

            for i in network.get_active_agents(k):
                neighborhood_avg = iop.sum([W[i, j] * q_j for j, q_j in i.received_messages.items()], dim=0)
                neighborhood_avg += W[i, i] * i.aux_vars["y"]
                i.aux_vars["y"] = neighborhood_avg + i.aux_vars["g_new"] - i.aux_vars["g"]
                i.aux_vars["g"] = i.aux_vars["g_new"]


SONATA = ATCTracking  # alias
NEXT = ATCTracking  # alias
ATCT = ATCTracking  # alias


@dataclass(eq=False)
class NIDS(DstAlgorithm):
    r"""
    NIDS [r10]_ gradient tracking algorithm characterized by the update steps below.

    .. math::
        \mathbf{x}_{i, k+1}
        = \sum_j \tilde{\mathbf{W}}_{ij} (2 x_{j,k} - x_{j, k-1}
        - \rho \nabla f_j(\mathbf{x}_{j,k}) + \rho \nabla f_j(\mathbf{x}_{j,k-1}))

    where
    :math:`\mathbf{x}_{i, k}` is agent i's local optimization variable at iteration k,
    :math:`\rho` is the step size,
    :math:`f_i` is agent i's local cost function,
    j is a neighbor of i or i itself,
    and :math:`\tilde{\mathbf{W}} = (\mathbf{I} + \mathbf{W}) / 2`
    with :math:`\mathbf{W}` are the Metropolis weights.

    This is a simplified version of the algorithm proposed in [r10]_ (see eq. (9) therein).

    .. [r10] Z. Li, W. Shi, and M. Yan, "A Decentralized Proximal-Gradient Method With Network
            Independent Step-Sizes and Separated Convergence Rates," IEEE Trans. Signal Process.,
            vol. 67, no. 17, pp. 4494-4506, Sep. 2019, doi: 10.1109/TSP.2019.2926022.

    """

    iterations: int
    step_size: float
    name: str = "NIDS"

    def run(self, network: Network) -> None:
        r"""
        Run the algorithm.

        Args:
            network: provides agents, neighbors etc.

        """
        # initialization (iteration k=0)
        for i in network.get_all_agents():
            x0 = np.zeros(i.cost_function.domain_shape)
            i.initialize(
                x=x0,
                received_msgs=dict.fromkeys(network.get_neighbors(i), x0),
                aux_vars={"x_old": x0, "g": x0, "g_old": x0, "y": x0},
            )

        W = iop.from_numpy_like(network.metropolis_weights, network.get_all_agents()[0].x)  # noqa: N806
        W_tilde = 0.5 * (iop.eye_like(W) + W)  # noqa: N806

        # first iteration (iteration k=1)
        for i in network.get_active_agents(0):
            i.aux_vars["g"] = i.cost_function.gradient(i.x)  # store grad f_i(x_0)
            i.x = i.aux_vars["x_old"] - self.step_size * i.aux_vars["g"]

        # subsequent iterations (k >= 2)
        for k in range(1, self.iterations):
            for i in network.get_active_agents(k):
                i.aux_vars["g_old"] = i.aux_vars["g"]  # store grad f_i(x_{k-1})
                i.aux_vars["g"] = i.cost_function.gradient(i.x)  # store grad f_i(x_k)
                i.aux_vars["y"] = (
                    2 * i.x
                    - i.aux_vars["x_old"]
                    - self.step_size * i.aux_vars["g"]
                    + self.step_size * i.aux_vars["g_old"]
                )
            for i in network.get_active_agents(k):
                network.broadcast(i, i.aux_vars["y"])
            for i in network.get_active_agents(k):
                network.receive_all(i)
            for i in network.get_active_agents(k):
                s = iop.stack(
                    [W_tilde[i, j] * y_j for j, y_j in i.received_messages.items()],
                    dim=0,
                )
                neighborhood_avg = iop.sum(
                    s,
                    dim=0,
                )
                neighborhood_avg += W_tilde[i, i] * i.aux_vars["y"]
                i.aux_vars["x_old"] = i.x  # store x_k
                i.x = neighborhood_avg  # update x_{k+1}


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
        Run the algorithm.

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
                i.x = i.cost_function.proximal(y=iop.sum(i.aux_vars["z"], dim=0) / pN[i], rho=1 / pN[i])
            for i in network.get_active_agents(k):
                for j in network.get_neighbors(i):
                    network.send(i, j, i.aux_vars["z"][j] - 2 * self.rho * i.x)
            for i in network.get_active_agents(k):
                network.receive_all(i)
            for i in network.get_active_agents(k):
                for j in network.get_neighbors(i):
                    i.aux_vars["z"][j] = (1 - self.alpha) * i.aux_vars["z"][j] - self.alpha * (i.received_messages[j])


@dataclass(eq=False)
class ATG(DstAlgorithm):
    r"""
    ADMM-Tracking Gradient (ATG) [r11]_ characterized by the update steps below.

    .. math::
        \begin{bmatrix} \mathbf{y}_{i,k} \\ \mathbf{s}_{i,k} \end{bmatrix} = \frac{1}{1 + \rho N_i}
        \left( \begin{bmatrix} \mathbf{x}_{i,k} \\ \nabla f_i(\mathbf{x}_{i,k}) \end{bmatrix}
        + \sum_j \mathbf{z}_{ij, k} \right)

    .. math::
        \mathbf{x}_{i,k+1} = (1 - \gamma) \mathbf{x}_{i,k}
        + \gamma \left( \mathbf{y}_{i,k} - \delta \mathbf{s}_{i,k} \right)

    .. math::
        \mathbf{z}_{ij, k+1} = (1-\alpha) \mathbf{z}_{ij, k} - \alpha \left( \mathbf{z}_{ji, k}
        - 2 \rho \begin{bmatrix} \mathbf{y}_{i,k} \\ \mathbf{s}_{i,k} \end{bmatrix} \right)

    where
    :math:`\mathbf{x}_{i, k} \in \mathbb{R}^n` is agent i's local optimization variable at iteration k,
    :math:`\mathbf{y}_{i,k}, \ \mathbf{s}_{i,k} \in \mathbb{R}^n`
    and :math:`\mathbf{z}_{ij,k} \in \mathbb{R}^{2n}` are auxiliary variables,
    :math:`N_i` is the number of neighbors of i,
    :math:`f_i` is i's local cost function,
    j is a neighbor of i. The parameters are: the penalty :math:`\rho > 0`, the relaxation :math:`\alpha \in (0, 1)`,
    the step-size :math:`\delta > 0`, and the mixing parameter :math:`\gamma > 0`. Notice that the convergence of
    the algorithm is guaranteed provided that :math:`\delta, \ \gamma` are below certain thresholds.

    The idea of the algorithm is to apply distributed ADMM to perform gradient tracking,
    instead of the usual average consensus.

    Aliases: :class:`ADMMTracking`, :class:`ADMMTrackingGradient`

    .. [r11] G. Carnevale, N. Bastianello, G. Notarstefano, and R. Carli, "ADMM-Tracking Gradient for Distributed
             Optimization Over Asynchronous and Unreliable Networks," IEEE Trans. Automat. Contr., vol. 70, no. 8,
             pp. 5160-5175, Aug. 2025, doi: 10.1109/TAC.2025.3539454.

    """

    iterations: int
    rho: float
    alpha: float
    gamma: float = 0.1
    delta: float = 0.1
    name: str = "ATG"

    def run(self, network: Network) -> None:
        r"""
        Run the algorithm.

        Args:
            network: provides agents, neighbors etc.

        """
        pN = {i: self.rho * len(network.get_neighbors(i)) for i in network.get_all_agents()}  # noqa: N806
        all_agents = network.get_all_agents()
        for i in all_agents:
            z_y0 = np.zeros((len(all_agents), *(i.cost_function.domain_shape)))
            z_s0 = np.zeros((len(all_agents), *(i.cost_function.domain_shape)))
            x0 = np.zeros(i.cost_function.domain_shape)
            i.initialize(
                x=x0,
                aux_vars={"y": x0, "s": x0, "z_y": z_y0, "z_s": z_s0},
                received_msgs=dict.fromkeys(network.get_neighbors(i), x0),
            )
        for k in range(self.iterations):
            # step 1: update consensus-ADMM variables
            for i in network.get_active_agents(k):
                # update auxiliary variables
                i.aux_vars["y"] = (i.x + iop.sum(i.aux_vars["z_y"], dim=0)) / (1 + pN[i])
                i.aux_vars["s"] = (i.cost_function.gradient(i.x) + iop.sum(i.aux_vars["z_s"], dim=0)) / (1 + pN[i])
                # update local state
                i.x = (1 - self.gamma) * i.x + self.gamma * (i.aux_vars["y"] - self.delta * i.aux_vars["s"])

            # step 2: communicate and update z_{ij} variables
            for i in network.get_active_agents(k):
                for j in network.get_neighbors(i):
                    # transmit the messages as a single message, stacking along the first axis
                    network.send(i, j,
                        iop.stack(
                            (
                                -i.aux_vars["z_y"][j] + 2 * self.rho * i.aux_vars["y"],
                                -i.aux_vars["z_s"][j] + 2 * self.rho * i.aux_vars["s"],
                            ), dim=0),
                    )  # fmt: skip
            for i in network.get_active_agents(k):
                network.receive_all(i)
            for i in network.get_active_agents(k):
                for j in network.get_neighbors(i):
                    i.aux_vars["z_y"][j] = (1 - self.alpha) * i.aux_vars["z_y"][j] \
                                         + self.alpha * i.received_messages[j][0]  # fmt: skip
                    i.aux_vars["z_s"][j] = (1 - self.alpha) * i.aux_vars["z_s"][j] \
                                         + self.alpha * i.received_messages[j][1]  # fmt: skip


ADMMTracking = ATG  # alias
ADMMTrackingGradient = ATG  # alias


@dataclass(eq=False)
class DLM(DstAlgorithm):
    r"""
    Decentralized Linearized ADMM (DLM) [r12]_ characterized by the update steps below (see also [r13]_).

    .. math::
        \mathbf{x}_{i,k+1} = \mathbf{x}_{i,k} - \mu \left( \nabla f_i(\mathbf{x}_{i,k})
        + \rho \sum_j (\mathbf{x}_{i,k} - \mathbf{x}_{j,k}) + \mathbf{z}_{i,k} \right)

    .. math::
        \mathbf{z}_{i, k+1} = \mathbf{z}_{i, k} + \rho \sum_j (\mathbf{x}_{i,k+1} - \mathbf{x}_{j,k+1})

    where
    :math:`\mathbf{x}_{i, k}` is agent i's local optimization variable at iteration k,
    :math:`\mathbf{z}_{i,k}` is the local dual variable,
    :math:`f_i` is i's local cost function,
    j is a neighbor of i. The parameters are: the penalty :math:`\rho > 0` and the step-size :math:`\mu > 0`.

    Alias: :class:`DecentralizedLinearizedADMM`

    .. [r12] Q. Ling, W. Shi, G. Wu, and A. Ribeiro, "DLM: Decentralized Linearized Alternating Direction
             Method of Multipliers," IEEE Transactions on Signal Processing, vol. 63, no. 15, pp. 4051-4064,
             Aug. 2015, doi: 10.1109/TSP.2015.2436358.

    .. [r13] S. A. Alghunaim, E. K. Ryu, K. Yuan, and A. H. Sayed, "Decentralized Proximal Gradient Algorithms
            With Linear Convergence Rates," IEEE Transactions on Automatic Control, vol. 66, no. 6, pp. 2787-2794,
            Jun. 2021, doi: 10.1109/TAC.2020.3009363.

    """

    iterations: int
    step_size: float
    penalty: float
    name: str = "DLM"

    def run(self, network: Network) -> None:
        r"""
        Run the algorithm.

        Args:
            network: provides agents, neighbors etc.

        """
        all_agents = network.get_all_agents()
        for i in all_agents:
            x0 = np.zeros(i.cost_function.domain_shape)
            i.initialize(
                x=x0,
                aux_vars={"y": np.zeros(i.cost_function.domain_shape)},  # y must be initialized to zero
            )

        # step 0: first communication round
        for i in network.get_active_agents(0):
            network.broadcast(i, i.x)
        for i in network.get_active_agents(0):
            network.receive_all(i)
        # compute and store \sum_j (\mathbf{x}_{i,0} - \mathbf{x}_{j,0})
        for i in network.get_active_agents(0):
            s = iop.stack([i.x - x_j for _, x_j in i.received_messages.items()], dim=0)
            i.aux_vars["s"] = iop.sum(s, dim=0)  # pyright: ignore[reportArgumentType]

        # main iteration
        for k in range(self.iterations):
            # step 1: update primal variable
            for i in network.get_active_agents(k):
                i.x = i.x - self.step_size * (  # noqa: PLR6104
                    i.cost_function.gradient(i.x) + self.penalty * i.aux_vars["s"] + i.aux_vars["y"]
                )

            # step 2: communication round
            for i in network.get_active_agents(k):
                network.broadcast(i, i.x)
            for i in network.get_active_agents(k):
                network.receive_all(i)
            # compute and store \sum_j (\mathbf{x}_{i,k+1} - \mathbf{x}_{j,k+1})
            for i in network.get_active_agents(k):
                s = iop.stack([i.x - x_j for _, x_j in i.received_messages.items()], dim=0)
                i.aux_vars["s"] = iop.sum(s, dim=0)  # pyright: ignore[reportArgumentType]

            # step 3: update dual variable
            for i in network.get_active_agents(k):
                i.aux_vars["y"] += self.penalty * i.aux_vars["s"]


DecentralizedLinearizedADMM = DLM  # alias
