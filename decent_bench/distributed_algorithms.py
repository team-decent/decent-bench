from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, final

import decent_bench.utils.algorithm_helpers as alg_helpers
import decent_bench.utils.interoperability as iop
from decent_bench.networks import P2PNetwork

if TYPE_CHECKING:
    from decent_bench.utils.array import Array


class Algorithm(ABC):
    """
    Decentralized algorithm.

    Agents collaborate to solve an optimization problem using peer-to-peer communication.
    """

    @property
    @abstractmethod
    def iterations(self) -> int:
        """Number of iterations to run the algorithm for."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the algorithm."""

    @abstractmethod
    def initialize(self, network: P2PNetwork) -> None:
        """
        Initialize the algorithm.

        Args:
            network: provides agents, neighbors etc.

        """

    @abstractmethod
    def step(self, network: P2PNetwork, iteration: int) -> None:
        """
        Perform one iteration of the algorithm.

        Args:
            network: provides agents, neighbors etc.
            iteration: current iteration number

        """

    def finalize(self, network: P2PNetwork) -> None:
        """
        Finalize the algorithm.

        Note:
            Override method as needed.
            Does not need to be implemented if no finalization is required.
            By default it is used to clean up auxiliary variables to free memory.

        Args:
            network: provides agents, neighbors etc.

        """
        for i in network.agents():
            if i.aux_vars is not None:
                i.aux_vars.clear()

    @final
    def run(
        self,
        network: P2PNetwork,
        start_iteration: int = 0,
        progress_callback: Callable[[int], None] | None = None,
    ) -> None:
        """
        Run the algorithm.

        Note:
            This method first calls :meth:`initialize`, then :meth:`step` for the specified number of iterations
            and finally :meth:`finalize`.

        Warning:
            Do not override this method. Instead, override :meth:`initialize`, :meth:`step` and :meth:`finalize`
            as needed.

        Args:
            network: provides agents, neighbors etc.
            start_iteration: iteration number to start from, used when resuming from a checkpoint.
            progress_callback: optional callback to report progress after each iteration. If greater than 0,
                :meth:`initialize` will be skipped.

        Raises:
            ValueError: if start_iteration is not in [0, iterations]

        """
        if start_iteration < 0 or start_iteration > self.iterations:
            raise ValueError(
                f"Invalid start_iteration {start_iteration} for algorithm with {self.iterations} iterations"
            )

        if start_iteration == 0:
            self.initialize(network)
        for k in range(start_iteration, self.iterations):
            self.step(network, k)
            if progress_callback is not None:
                progress_callback(k)
        self.finalize(network)


@dataclass(eq=False)
class DGD(Algorithm):
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

    step_size: float
    x0: "Array | None" = None
    iterations: int = 100
    name: str = "DGD"

    def initialize(self, network: P2PNetwork) -> None:  # noqa: D102
        self.x0 = alg_helpers.zero_initialization(self.x0, network)
        for i in network.agents():
            i.initialize(x=self.x0, received_msgs=dict.fromkeys(network.neighbors(i), self.x0))

        self.W = network.weights

    def step(self, network: P2PNetwork, iteration: int) -> None:  # noqa: D102
        for i in network.active_agents(iteration):
            s = iop.stack([self.W[i, j] * x_j for j, x_j in i.messages.items()])
            neighborhood_avg = iop.sum(s, dim=0)
            neighborhood_avg += self.W[i, i] * i.x
            i.x = neighborhood_avg - self.step_size * i.cost.gradient(i.x)

        for i in network.active_agents(iteration):
            network.broadcast(i, i.x)

        for i in network.active_agents(iteration):
            network.receive_all(i)


@dataclass(eq=False)
class ATC(Algorithm):
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

    step_size: float
    x0: "Array | None" = None
    iterations: int = 100
    name: str = "ATC"

    def initialize(self, network: P2PNetwork) -> None:  # noqa: D102
        self.x0 = alg_helpers.zero_initialization(self.x0, network)
        for i in network.agents():
            i.initialize(
                x=self.x0,
                received_msgs=dict.fromkeys(network.neighbors(i), self.x0),
                aux_vars={"y": self.x0},
            )

        self.W = network.weights

    def step(self, network: P2PNetwork, iteration: int) -> None:  # noqa: D102
        # gradient step (a.k.a. adapt step)
        for i in network.active_agents(iteration):
            i.aux_vars["y"] = i.x - self.step_size * i.cost.gradient(i.x)

        # transmit and receive
        for i in network.active_agents(iteration):
            network.broadcast(i, i.aux_vars["y"])

        # consensus (a.k.a. combine step)
        for i in network.active_agents(iteration):
            network.receive_all(i)

        for i in network.active_agents(iteration):
            s = iop.stack([self.W[i, j] * x_j for j, x_j in i.messages.items()], dim=0)
            neighborhood_avg = iop.sum(s, dim=0)
            neighborhood_avg += self.W[i, i] * i.x
            i.x = neighborhood_avg


AdaptThenCombine = ATC  # alias


@dataclass(eq=False)
class SimpleGT(Algorithm):
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

    Alias: :class:`SimpleGradientTracking`

    """

    step_size: float
    x0: "Array | None" = None
    iterations: int = 100
    name: str = "SimpleGT"

    def initialize(self, network: P2PNetwork) -> None:  # noqa: D102
        self.x0 = alg_helpers.zero_initialization(self.x0, network)
        for i in network.agents():
            y0 = iop.zeros(framework=i.cost.framework, shape=i.cost.shape, device=i.cost.device)
            neighbors = network.neighbors(i)
            i.initialize(x=self.x0, received_msgs=dict.fromkeys(neighbors, self.x0), aux_vars={"y": y0})

        self.W = network.weights

    def step(self, network: P2PNetwork, iteration: int) -> None:  # noqa: D102
        for i in network.active_agents(iteration):
            i.aux_vars["y_new"] = i.x - self.step_size * i.cost.gradient(i.x)
            s = iop.stack([self.W[i, j] * x_j for j, x_j in i.messages.items()])
            neighborhood_avg = iop.sum(s, dim=0)
            neighborhood_avg += self.W[i, i] * i.x
            i.x = i.aux_vars["y_new"] - i.aux_vars["y"] + neighborhood_avg
            i.aux_vars["y"] = i.aux_vars["y_new"]

        for i in network.active_agents(iteration):
            network.broadcast(i, i.x)

        for i in network.active_agents(iteration):
            network.receive_all(i)


SimpleGradientTracking = SimpleGT  # Alias


@dataclass(eq=False)
class ED(Algorithm):
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

    Alias: :class:`ExactDiffusion`

    """

    step_size: float
    x0: "Array | None" = None
    iterations: int = 100
    name: str = "ED"

    def initialize(self, network: P2PNetwork) -> None:  # noqa: D102
        self.x0 = alg_helpers.zero_initialization(self.x0, network)
        for i in network.agents():
            y0 = iop.zeros(framework=i.cost.framework, shape=i.cost.shape, device=i.cost.device)
            y1 = self.x0 - self.step_size * i.cost.gradient(self.x0)
            # note: msg0's y1 is an approximation of the neighbors' y1 (x0 and y0 are exact: all agents start with same)
            msg0 = self.x0 + y1 - y0
            i.initialize(
                x=self.x0,
                aux_vars={"y": y0, "y_new": y1},
                received_msgs=dict.fromkeys(network.neighbors(i), msg0),
            )

        self.W = network.weights
        self.W = 0.5 * (iop.eye_like(self.W) + self.W)

    def step(self, network: P2PNetwork, iteration: int) -> None:  # noqa: D102
        for i in network.active_agents(iteration):
            s = iop.stack([self.W[i, j] * msg for j, msg in i.messages.items()])
            i.x = iop.sum(s, dim=0) + self.W[i, i] * (i.x + i.aux_vars["y_new"] - i.aux_vars["y"])
            i.aux_vars["y"] = i.aux_vars["y_new"]
            i.aux_vars["y_new"] = i.x - self.step_size * i.cost.gradient(i.x)

        for i in network.active_agents(iteration):
            network.broadcast(i, i.x + i.aux_vars["y_new"] - i.aux_vars["y"])

        for i in network.active_agents(iteration):
            network.receive_all(i)


ExactDiffusion = ED  # alias


@dataclass(eq=False)
class AugDGM(Algorithm):
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

    step_size: float
    x0: "Array | None" = None
    iterations: int = 100
    name: str = "Aug-DGM"

    def initialize(self, network: P2PNetwork) -> None:  # noqa: D102
        self.x0 = alg_helpers.zero_initialization(self.x0, network)
        for i in network.agents():
            y0 = i.cost.gradient(self.x0)
            neighbors = network.neighbors(i)
            i.initialize(
                x=self.x0,
                received_msgs=dict.fromkeys(neighbors, self.x0),
                aux_vars={"y": y0, "g": y0, "g_new": self.x0, "s": self.x0},
            )

        self.W = network.weights

    def step(self, network: P2PNetwork, iteration: int) -> None:  # noqa: D102
        # 1st communication round
        #     step 1: perform local gradient step and communicate
        for i in network.active_agents(iteration):
            i.aux_vars["s"] = i.x - self.step_size * i.aux_vars["y"]

        for i in network.active_agents(iteration):
            network.broadcast(i, i.aux_vars["s"])

        #     step 2: update state and compute new local gradient
        for i in network.active_agents(iteration):
            network.receive_all(i)

        for i in network.active_agents(iteration):
            s = iop.stack([self.W[i, j] * s_j for j, s_j in i.messages.items()])
            neighborhood_avg = iop.sum(s, dim=0)
            neighborhood_avg += self.W[i, i] * i.aux_vars["s"]
            i.x = neighborhood_avg
            i.aux_vars["g_new"] = i.cost.gradient(i.x)

        # 2nd communication round
        #     step 1: transmit local gradient tracker
        for i in network.active_agents(iteration):
            network.broadcast(i, i.aux_vars["y"] + i.aux_vars["g_new"] - i.aux_vars["g"])

        #     step 2: update y (global gradient estimator)
        for i in network.active_agents(iteration):
            network.receive_all(i)

        for i in network.active_agents(iteration):
            s = iop.stack([self.W[i, j] * q_j for j, q_j in i.messages.items()])
            neighborhood_avg = iop.sum(s, dim=0)
            neighborhood_avg += self.W[i, i] * (i.aux_vars["y"] + i.aux_vars["g_new"] - i.aux_vars["g"])
            i.aux_vars["y"] = neighborhood_avg
            i.aux_vars["g"] = i.aux_vars["g_new"]


ATCDIGing = AugDGM  # alias


@dataclass(eq=False)
class WangElia(Algorithm):
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

    step_size: float
    x0: "Array | None" = None
    iterations: int = 100
    name: str = "Wang-Elia"

    def initialize(self, network: P2PNetwork) -> None:  # noqa: D102
        self.x0 = alg_helpers.zero_initialization(self.x0, network)
        for i in network.agents():
            neighbors = network.neighbors(i)
            i.initialize(
                x=self.x0,
                received_msgs=dict.fromkeys(neighbors, self.x0),
                aux_vars={"z": self.x0, "x_old": self.x0},
            )

        W = network.weights  # noqa: N806
        K = 0.5 * (iop.eye_like(W) - W)  # noqa: N806

        self.K = K

    def step(self, network: P2PNetwork, iteration: int) -> None:  # noqa: D102
        # 1st communication round
        for i in network.active_agents(iteration):
            network.broadcast(i, i.x + i.aux_vars["z"])

        # do consensus and local gradient step
        for i in network.active_agents(iteration):
            network.receive_all(i)

        for i in network.active_agents(iteration):
            s = iop.stack([self.K[i, j] * m_j for j, m_j in i.messages.items()])
            neighborhood_avg = iop.sum(s, dim=0)
            neighborhood_avg += self.K[i, i] * (i.x + i.aux_vars["z"])

            i.aux_vars["x_old"] = i.x
            i.x = i.x - neighborhood_avg - self.step_size * i.cost.gradient(i.x)

        # 2nd communication round
        for i in network.active_agents(iteration):
            network.broadcast(i, i.aux_vars["x_old"])

        # update auxiliary variable
        for i in network.active_agents(iteration):
            network.receive_all(i)

        for i in network.active_agents(iteration):
            s = iop.stack([self.K[i, j] * m_j for j, m_j in i.messages.items()])
            neighborhood_avg = iop.sum(s, dim=0)
            neighborhood_avg += self.K[i, i] * i.aux_vars["x_old"]
            i.aux_vars["z"] += neighborhood_avg


@dataclass(eq=False)
class EXTRA(Algorithm):
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

    step_size: float
    x0: "Array | None" = None
    iterations: int = 100
    name: str = "EXTRA"

    def initialize(self, network: P2PNetwork) -> None:  # noqa: D102
        self.x0 = alg_helpers.zero_initialization(self.x0, network)
        for i in network.agents():
            i.initialize(
                x=self.x0,
                received_msgs=dict.fromkeys(network.neighbors(i), self.x0),
                aux_vars={"x_old": self.x0, "x_old_old": self.x0, "x_cons": self.x0},
            )

        self.W = network.weights

    def step(self, network: P2PNetwork, iteration: int) -> None:  # noqa: D102
        if iteration == 0:
            # first iteration (iteration k=1)
            for i in network.active_agents(0):
                network.broadcast(i, i.x)

            for i in network.active_agents(0):
                network.receive_all(i)

            for i in network.active_agents(0):
                s = iop.stack([self.W[i, j] * x_j for j, x_j in i.messages.items()])
                neighborhood_avg = iop.sum(s, dim=0)
                neighborhood_avg += self.W[i, i] * i.x
                i.aux_vars["x_cons"] = neighborhood_avg  # store W x_k
                i.aux_vars["x_old"] = i.x  # store x_0
                i.x = neighborhood_avg - self.step_size * i.cost.gradient(i.x)
        else:
            # subsequent iterations (k >= 2)
            for i in network.active_agents(iteration):
                network.broadcast(i, i.x)

            for i in network.active_agents(iteration):
                network.receive_all(i)

            for i in network.active_agents(iteration):
                s = iop.stack([self.W[i, j] * x_j for j, x_j in i.messages.items()])
                neighborhood_avg = iop.sum(s, dim=0)
                neighborhood_avg += self.W[i, i] * i.x
                i.aux_vars["x_old_old"] = i.aux_vars["x_old"]  # store x_{k-1}
                i.aux_vars["x_old"] = i.x  # store x_k
                # update x_{k+1}
                i.x = (
                    i.x
                    + neighborhood_avg
                    - 0.5 * i.aux_vars["x_old_old"]
                    - 0.5 * i.aux_vars["x_cons"]
                    - self.step_size * (i.cost.gradient(i.x) - i.cost.gradient(i.aux_vars["x_old_old"]))
                )
                i.aux_vars["x_cons"] = neighborhood_avg  # store W x_k


@dataclass(eq=False)
class ATCTracking(Algorithm):
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

    step_size: float
    x0: "Array | None" = None
    iterations: int = 100
    name: str = "ATC-Tracking"

    def initialize(self, network: P2PNetwork) -> None:  # noqa: D102
        self.x0 = alg_helpers.zero_initialization(self.x0, network)
        for i in network.agents():
            y0 = i.cost.gradient(self.x0)
            neighbors = network.neighbors(i)
            i.initialize(
                x=self.x0,
                received_msgs=dict.fromkeys(neighbors, self.x0),
                aux_vars={"y": y0, "g": y0, "g_new": self.x0, "s": self.x0},
            )

        self.W = network.weights

    def step(self, network: P2PNetwork, iteration: int) -> None:  # noqa: D102
        # 1st communication round
        #     step 1: perform local gradient step and communicate
        for i in network.active_agents(iteration):
            i.aux_vars["s"] = i.x - self.step_size * i.aux_vars["y"]

        for i in network.active_agents(iteration):
            network.broadcast(i, i.aux_vars["s"])

        #     step 2: update state and compute new local gradient
        for i in network.active_agents(iteration):
            network.receive_all(i)

        for i in network.active_agents(iteration):
            s = iop.stack([self.W[i, j] * s_j for j, s_j in i.messages.items()])
            neighborhood_avg = iop.sum(s, dim=0)
            neighborhood_avg += self.W[i, i] * i.aux_vars["s"]
            i.x = neighborhood_avg
            i.aux_vars["g_new"] = i.cost.gradient(i.x)

        # 2nd communication round
        #     step 1: transmit local gradient tracker
        for i in network.active_agents(iteration):
            network.broadcast(i, i.aux_vars["y"])

        #     step 2: update y (global gradient estimator)
        for i in network.active_agents(iteration):
            network.receive_all(i)

        for i in network.active_agents(iteration):
            s = iop.stack([self.W[i, j] * q_j for j, q_j in i.messages.items()])
            neighborhood_avg = iop.sum(s, dim=0)
            neighborhood_avg += self.W[i, i] * i.aux_vars["y"]
            i.aux_vars["y"] = neighborhood_avg + i.aux_vars["g_new"] - i.aux_vars["g"]
            i.aux_vars["g"] = i.aux_vars["g_new"]


SONATA = ATCTracking  # alias
NEXT = ATCTracking  # alias
ATCT = ATCTracking  # alias


@dataclass(eq=False)
class NIDS(Algorithm):
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

    step_size: float
    x0: "Array | None" = None
    iterations: int = 100
    name: str = "NIDS"

    def initialize(self, network: P2PNetwork) -> None:  # noqa: D102
        self.x0 = alg_helpers.zero_initialization(self.x0, network)
        for i in network.agents():
            i.initialize(
                x=self.x0,
                received_msgs=dict.fromkeys(network.neighbors(i), self.x0),
                aux_vars={"x_old": self.x0, "g": self.x0, "g_old": self.x0, "y": self.x0},
            )

        W = network.weights  # noqa: N806
        W_tilde = 0.5 * (iop.eye_like(W) + W)  # noqa: N806
        self.W_tilde = W_tilde

    def step(self, network: P2PNetwork, iteration: int) -> None:  # noqa: D102
        if iteration == 0:
            # first iteration (iteration k=1)
            for i in network.active_agents(0):
                i.aux_vars["g"] = i.cost.gradient(i.x)  # store grad f_i(x_0)
                i.x = i.aux_vars["x_old"] - self.step_size * i.aux_vars["g"]
        else:
            # subsequent iterations (k >= 2)
            for i in network.active_agents(iteration):
                i.aux_vars["g_old"] = i.aux_vars["g"]  # store grad f_i(x_{k-1})
                i.aux_vars["g"] = i.cost.gradient(i.x)  # store grad f_i(x_k)
                i.aux_vars["y"] = (
                    2 * i.x
                    - i.aux_vars["x_old"]
                    - self.step_size * i.aux_vars["g"]
                    + self.step_size * i.aux_vars["g_old"]
                )
            for i in network.active_agents(iteration):
                network.broadcast(i, i.aux_vars["y"])
            for i in network.active_agents(iteration):
                network.receive_all(i)
            for i in network.active_agents(iteration):
                s = iop.stack([self.W_tilde[i, j] * y_j for j, y_j in i.messages.items()])
                neighborhood_avg = iop.sum(s, dim=0)
                neighborhood_avg += self.W_tilde[i, i] * i.aux_vars["y"]
                i.aux_vars["x_old"] = i.x  # store x_k
                i.x = neighborhood_avg  # update x_{k+1}


@dataclass(eq=False)
class ADMM(Algorithm):
    r"""
    Distributed Alternating Direction Method of Multipliers characterized by the update step below.

    .. math::
        \mathbf{x}_{i, k+1} = \operatorname{prox}_{\frac{1}{\rho N_i} f_i}
        \left(\sum_j \mathbf{Z}_{ij, k} \frac{1}{\rho N_i} \right)
    .. math::
        \mathbf{Z}_{ij, k+1} = (1-\alpha) \mathbf{Z}_{ij, k} - \alpha (\mathbf{Z}_{ji, k} - 2 \rho \mathbf{x}_{j, k+1})

    where
    :math:`\mathbf{x}_{i, k}` is agent i's local optimization variable at iteration k,
    :math:`\operatorname{prox}` is the proximal operator described in :meth:`Cost.proximal()
    <decent_bench.costs.Cost.proximal>`,
    :math:`\rho > 0` is the Lagrangian penalty parameter,
    :math:`N_i` is the number of neighbors of i,
    :math:`f_i` is i's local cost function,
    j is a neighbor of i,
    and :math:`\alpha \in (0, 1)` is the relaxation parameter.

    Note:
        ``z0`` is of shape :attr:`agent.cost.shape <decent_bench.costs.Cost.shape>` which is then stacked for all
        agents to form ``z`` of shape ``(num_agents, *agent.cost.shape)``.

    """

    rho: float
    alpha: float
    z0: "Array | None" = None
    iterations: int = 100
    name: str = "ADMM"

    def initialize(self, network: P2PNetwork) -> None:  # noqa: D102
        pN = {i: self.rho * len(network.neighbors(i)) for i in network.agents()}  # noqa: N806
        all_agents = network.agents()
        self.z0 = alg_helpers.zero_initialization(self.z0, network, stacked_copies=len(all_agents))
        for i in all_agents:
            x1 = i.cost.proximal(x=iop.sum(self.z0, dim=0) / pN[i], rho=1 / pN[i])
            # note: msg0's x1 is an approximation of the neighbors' x1 (z0 is exact: all agents start with same)
            msg0 = self.z0[i] - 2 * self.rho * x1
            i.initialize(
                x=x1,
                aux_vars={"z": self.z0},
                received_msgs=dict.fromkeys(network.neighbors(i), msg0),
            )

        self.pN = pN

    def step(self, network: P2PNetwork, iteration: int) -> None:  # noqa: D102
        for i in network.active_agents(iteration):
            i.x = i.cost.proximal(x=iop.sum(i.aux_vars["z"], dim=0) / self.pN[i], rho=1 / self.pN[i])

        for i in network.active_agents(iteration):
            for j in network.neighbors(i):
                network.send(i, j, i.aux_vars["z"][j] - 2 * self.rho * i.x)

        for i in network.active_agents(iteration):
            network.receive_all(i)

        for i in network.active_agents(iteration):
            for j in network.neighbors(i):
                i.aux_vars["z"][j] = (1 - self.alpha) * i.aux_vars["z"][j] - self.alpha * (i.messages[j])


@dataclass(eq=False)
class ATG(Algorithm):
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

    rho: float
    alpha: float
    gamma: float = 0.1
    delta: float = 0.1
    x0: "Array | None" = None
    iterations: int = 100
    name: str = "ATG"

    def initialize(self, network: P2PNetwork) -> None:  # noqa: D102
        pN = {i: self.rho * len(network.neighbors(i)) for i in network.agents()}  # noqa: N806
        all_agents = network.agents()
        self.x0 = alg_helpers.zero_initialization(self.x0, network)
        for i in all_agents:
            z_y0 = iop.zeros(
                framework=i.cost.framework,
                shape=(len(all_agents), *(i.cost.shape)),
                device=i.cost.device,
            )
            z_s0 = iop.zeros(
                framework=i.cost.framework,
                shape=(len(all_agents), *(i.cost.shape)),
                device=i.cost.device,
            )
            i.initialize(
                x=self.x0,
                aux_vars={"y": self.x0, "s": self.x0, "z_y": z_y0, "z_s": z_s0},
                received_msgs=dict.fromkeys(network.neighbors(i), self.x0),
            )

        self.pN = pN

    def step(self, network: P2PNetwork, iteration: int) -> None:  # noqa: D102
        # step 1: update consensus-ADMM variables
        for i in network.active_agents(iteration):
            # update auxiliary variables
            i.aux_vars["y"] = (i.x + iop.sum(i.aux_vars["z_y"], dim=0)) / (1 + self.pN[i])
            i.aux_vars["s"] = (i.cost.gradient(i.x) + iop.sum(i.aux_vars["z_s"], dim=0)) / (1 + self.pN[i])
            # update local state
            i.x = (1 - self.gamma) * i.x + self.gamma * (i.aux_vars["y"] - self.delta * i.aux_vars["s"])

        # step 2: communicate and update z_{ij} variables
        for i in network.active_agents(iteration):
            for j in network.neighbors(i):
                # transmit the messages as a single message, stacking along the first axis
                s = iop.stack(
                    (
                        -i.aux_vars["z_y"][j] + 2 * self.rho * i.aux_vars["y"],
                        -i.aux_vars["z_s"][j] + 2 * self.rho * i.aux_vars["s"],
                    ),
                    dim=0,
                )
                network.send(i, j, s)
        for i in network.active_agents(iteration):
            network.receive_all(i)
        for i in network.active_agents(iteration):
            for j in network.neighbors(i):
                i.aux_vars["z_y"][j] = (1 - self.alpha) * i.aux_vars["z_y"][j] \
                                        + self.alpha * i.messages[j][0]  # fmt: skip
                i.aux_vars["z_s"][j] = (1 - self.alpha) * i.aux_vars["z_s"][j] \
                                        + self.alpha * i.messages[j][1]  # fmt: skip


ADMMTracking = ATG  # alias
ADMMTrackingGradient = ATG  # alias


@dataclass(eq=False)
class DLM(Algorithm):
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

    step_size: float
    penalty: float
    x0: "Array | None" = None
    iterations: int = 100
    name: str = "DLM"

    def initialize(self, network: P2PNetwork) -> None:  # noqa: D102
        self.x0 = alg_helpers.zero_initialization(self.x0, network)
        for i in network.agents():
            # y must be initialized to zero
            y = iop.zeros(framework=i.cost.framework, shape=i.cost.shape, device=i.cost.device)
            i.initialize(x=self.x0, aux_vars={"y": y})

    def step(self, network: P2PNetwork, iteration: int) -> None:  # noqa: D102
        if iteration == 0:
            # step 0: first communication round
            for i in network.active_agents(0):
                network.broadcast(i, i.x)

            for i in network.active_agents(0):
                network.receive_all(i)

            # compute and store \sum_j (\mathbf{x}_{i,0} - \mathbf{x}_{j,0})
            for i in network.active_agents(0):
                s = iop.stack([i.x - x_j for _, x_j in i.messages.items()])
                i.aux_vars["s"] = iop.sum(s, dim=0)  # pyright: ignore[reportArgumentType]
        else:
            # step 1: update primal variable
            for i in network.active_agents(iteration):
                i.x = i.x - self.step_size * (  # noqa: PLR6104
                    i.cost.gradient(i.x) + self.penalty * i.aux_vars["s"] + i.aux_vars["y"]
                )

            # step 2: communication round
            for i in network.active_agents(iteration):
                network.broadcast(i, i.x)

            for i in network.active_agents(iteration):
                network.receive_all(i)

            # compute and store \sum_j (\mathbf{x}_{i,k+1} - \mathbf{x}_{j,k+1})
            for i in network.active_agents(iteration):
                s = iop.stack([i.x - x_j for _, x_j in i.messages.items()])
                i.aux_vars["s"] = iop.sum(s, dim=0)  # pyright: ignore[reportArgumentType]

            # step 3: update dual variable
            for i in network.active_agents(iteration):
                i.aux_vars["y"] += self.penalty * i.aux_vars["s"]


DecentralizedLinearizedADMM = DLM  # alias
