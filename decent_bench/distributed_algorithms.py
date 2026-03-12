import random
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, cast, final

import decent_bench.utils.algorithm_helpers as alg_helpers
import decent_bench.utils.interoperability as iop
from decent_bench.costs import EmpiricalRiskCost
from decent_bench.networks import FedNetwork, Network, P2PNetwork
from decent_bench.schemes import ClientSelectionScheme, UniformClientSelection
from decent_bench.utils.types import ClientWeights

if TYPE_CHECKING:
    from decent_bench.agents import Agent
    from decent_bench.utils.array import Array


class Algorithm[NetworkT: Network](ABC):
    """Base class for decentralized algorithms."""

    @abstractmethod
    def __post_init__(self) -> None:
        """Optional hook to be called by dataclasses after __init__."""  # noqa: D401

    def __init_subclass__(cls, **kwargs: dict[str, Any]) -> None:
        """Validate `iterations` for all subclasses."""
        super().__init_subclass__(**kwargs)

        # override __post_init__ to inject `iterations` validation
        original_post_init: Callable[[Algorithm[NetworkT]], None] | None = getattr(cls, "__post_init__", None)

        def __post_init__(self: "Algorithm[NetworkT]") -> None:  # noqa: N807
            # inject `iterations` validation
            if self.iterations <= 0:
                raise ValueError("`iterations` must be positive")

            # add subclass's __post_init__ if any
            if original_post_init:
                original_post_init(self)

        setattr(cls, "__post_init__", __post_init__)  # noqa: B010

    @property
    @abstractmethod
    def iterations(self) -> int:
        """Number of iterations to run the algorithm for."""

    @iterations.setter
    @abstractmethod
    def iterations(self, value: int) -> None:
        """Set the number of iterations to run the algorithm for."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the algorithm."""

    @abstractmethod
    def initialize(self, network: NetworkT) -> None:
        """
        Initialize the algorithm.

        Args:
            network: provides the agents and topology for this algorithm.

        """

    @abstractmethod
    def step(self, network: NetworkT, iteration: int) -> None:
        """
        Perform one iteration of the algorithm.

        Args:
            network: provides the agents and topology for this algorithm.
            iteration: current iteration number.

        """

    @abstractmethod
    def _finalize_agents(self, network: NetworkT) -> Iterable["Agent"]:
        """
        Return the agents whose auxiliary variables should be cleared.

        Args:
            network: provides the agents and topology for this algorithm.

        """

    def finalize(self, network: NetworkT) -> None:
        """
        Finalize the algorithm.

        Note:
            Override :meth:`~decent_bench.distributed_algorithms.Algorithm._finalize_agents` to control which
            agents are finalized.

        Args:
            network: provides the agents and topology for this algorithm.

        """
        for agent in self._finalize_agents(network):
            if agent.aux_vars is not None:
                agent.aux_vars.clear()

    @final
    def _snapshot_agents(self, network: P2PNetwork, iteration: int) -> None:
        for i in network.agents():
            # Forcefully save a snapshot on the final iteration
            i.snapshot(iteration=iteration, force=iteration == self.iterations)

    @final
    def run(
        self,
        network: NetworkT,
        start_iteration: int = 0,
        progress_callback: Callable[[int], None] | None = None,
        skip_finalize: bool = False,
    ) -> None:
        """
        Run the algorithm.

        This method first calls :meth:`initialize`, then :meth:`step` for the specified number of iterations
        and finally :meth:`finalize`.

        Args:
            network: provides the agents and topology for this algorithm.
            start_iteration: iteration number to start from, used when resuming from a checkpoint. If greater than 0,
                :meth:`initialize` will be skipped.
            progress_callback: optional callback to report progress after each iteration.
            skip_finalize: if True, skip calling :meth:`finalize` after running the iterations. This is needed
                so that full agent states and training variables can be saved in checkpoints without being cleared
                by :meth:`finalize`.

        Raises:
            ValueError: if start_iteration is not in [0, iterations]

        Warning:
            Do not override this method. Instead, override :meth:`initialize`, :meth:`step` and :meth:`finalize`
            as needed.

        Note:
            The algorithm saves the agents' states every :attr:`~decent_bench.agents.Agent.state_snapshot_period`,
            by calling :meth:~decent_bench.agents.Agent.snapshot for each agent.

        """
        if start_iteration < 0 or start_iteration > self.iterations:
            raise ValueError(
                f"Invalid start_iteration {start_iteration} for algorithm with {self.iterations} iterations"
            )

        if start_iteration == 0:
            self.initialize(network)
        for k in range(start_iteration, self.iterations):
            self.step(network, k)
            # Already completed the iteration, so snapshot with k+1 to indicate the state after iteration k
            self._snapshot_agents(network, k + 1)  # type: ignore[arg-type]
            if progress_callback is not None:
                progress_callback(k)

        if not skip_finalize:
            self.finalize(network)


class P2PAlgorithm(Algorithm[P2PNetwork]):
    """Distributed algorithm - agents collaborate using peer-to-peer communication."""

    def _finalize_agents(self, network: P2PNetwork) -> Iterable["Agent"]:
        return network.agents()


class FedAlgorithm(Algorithm[FedNetwork]):
    r"""
    Federated algorithm - clients collaborate via a central server.

    Note:
        ``client_weights`` only affects how updates are aggregated at the server; it does not change the objective
        function being optimized (the goal is still to solve :math:`\min \sum_i f_i(x)`). To optimize a weighted
        objective :math:`\min \sum_i w_i f_i(x)`, scale each client's cost by ``w_i`` in the problem definition.

    """

    selection_scheme: ClientSelectionScheme | None = None
    _DEFAULT_SELECTION_SCHEME: Final[object] = object()
    client_weights: ClientWeights | None = None
    _DEFAULT_CLIENT_WEIGHTS: Final[object] = object()

    def _finalize_agents(self, network: FedNetwork) -> Iterable["Agent"]:
        return [network.server, *network.clients]

    def finalize(self, network: FedNetwork) -> None:
        """
        Finalize the algorithm and sync the final server model to clients.

        This performs one last server-to-client send/receive so that clients store the final model.
        """
        if network.clients:
            network.broadcast(msg=network.server.x)
            for client in network.clients:
                network.receive(receiver=client, sender=network.server)
                if network.server in client.messages:
                    client.x = iop.copy(client.messages[network.server])

        super().finalize(network)

    def select_clients(
        self,
        clients: Sequence["Agent"],
        iteration: int,
        selection_scheme: ClientSelectionScheme | object | None = _DEFAULT_SELECTION_SCHEME,
    ) -> list["Agent"]:
        """
        Select participating clients from an eligible pool.

        Args:
            clients: eligible clients to select from.
            iteration: current round index.
            selection_scheme: optional override for this call. If omitted, uses ``self.selection_scheme``.
                Pass ``None`` to force selecting all clients.

        """
        if selection_scheme is self._DEFAULT_SELECTION_SCHEME:
            selection_scheme = self.selection_scheme
        if selection_scheme is None:
            return list(clients)
        scheme = cast("ClientSelectionScheme", selection_scheme)
        return scheme.select(clients, iteration)

    @classmethod
    def _weights_for_clients(
        cls,
        clients: Sequence["Agent"],
        client_weights: ClientWeights | None,
    ) -> list[float]:
        if client_weights is None:
            weights = [alg_helpers.infer_client_weight(client) for client in clients]
        elif isinstance(client_weights, dict):
            weights = []
            for client in clients:
                if client.id not in client_weights:
                    raise ValueError(f"Missing weight for client id {client.id}")
                weights.append(float(client_weights[client.id]))
        else:
            max_id = max(client.id for client in clients)
            if len(client_weights) <= max_id:
                raise ValueError("client_weights sequence must be indexed by client id")
            weights = [float(client_weights[client.id]) for client in clients]
        if any(weight < 0 for weight in weights):
            raise ValueError("Client weights must be non-negative")
        return weights

    def aggregate(
        self,
        network: FedNetwork,
        selected_clients: Sequence["Agent"],
        client_weights: ClientWeights | object | None = _DEFAULT_CLIENT_WEIGHTS,
    ) -> None:
        """
        Aggregate client updates at the server.

        By default, this performs a weighted average of the received client models. If ``client_weights`` is not
        provided, ``self.client_weights`` is used. If weights are ``None``, they are inferred from client data size.

        Override this method for custom aggregation strategies (e.g., robust aggregation).

        Raises:
            ValueError: if the sum of client weights is non-positive.

        """
        if client_weights is self._DEFAULT_CLIENT_WEIGHTS:
            client_weights = self.client_weights

        network.receive(receiver=network.server, sender=selected_clients)
        received_clients = [client for client in selected_clients if client in network.server.messages]
        if not received_clients:
            return
        updates = [network.server.messages[client] for client in received_clients]
        weights = self._weights_for_clients(received_clients, cast("ClientWeights | None", client_weights))
        total_weight = sum(weights)
        if total_weight <= 0:
            raise ValueError("Sum of client weights must be positive")
        weighted_updates = [update * weight for update, weight in zip(updates, weights, strict=True)]
        network.server.x = iop.sum(iop.stack(weighted_updates, dim=0), dim=0) / total_weight

    def _selected_clients_for_round(self, network: FedNetwork, iteration: int) -> list["Agent"]:
        active_clients = network.active_clients(iteration)
        if not active_clients:
            return []
        return self.select_clients(active_clients, iteration)


@dataclass(eq=False)
class FedAvg(FedAlgorithm):
    r"""
    Federated Averaging (FedAvg) with local SGD epochs.

    .. math::
        \mathbf{x}_{i, k}^{(t+1)} = \mathbf{x}_{i, k}^{(t)} - \eta \nabla f_i(\mathbf{x}_{i, k}^{(t)})

    .. math::
        \mathbf{x}_{k+1} = \frac{1}{|S_k|} \sum_{i \in S_k} \mathbf{x}_{i, k}^{(E)}

    where :math:`t` indexes the local training epochs on each client and :math:`E` is the number of local epochs per
    round (``num_local_epochs``), :math:`\eta` is the step size, and :math:`S_k` is the set of participating clients at
    round :math:`k`. In FedAvg, each selected client performs ``num_local_epochs`` local SGD epochs, then the server
    aggregates the final local models to form :math:`\mathbf{x}_{k+1}`. The aggregation uses client weights, defaulting
    to data-size weights when ``client_weights`` is not provided. Client selection (subsampling) defaults to uniform
    sampling with fraction 1.0 (all active clients) and can be customized via ``selection_scheme``. For
    :class:`~decent_bench.costs.EmpiricalRiskCost`, local updates use mini-batches of size
    :attr:`EmpiricalRiskCost.batch_size <decent_bench.costs.EmpiricalRiskCost.batch_size>`; for generic costs, local
    updates use full-batch gradients.
    """

    # C=0.1; batch size= inf/10/50 (dataset sizes are bigger; normally 1/10 of the total dataset).
    # E= 5/20 (num local epochs).
    iterations: int = 100
    step_size: float = 0.001
    num_local_epochs: int = 1
    sgd_seed: int | None = None
    client_weights: ClientWeights | None = None
    selection_scheme: ClientSelectionScheme | None = field(
        default_factory=lambda: UniformClientSelection(client_fraction=1.0)
    )
    x0: "Array | None" = None
    _sgd_rngs: dict[int, random.Random] | None = field(init=False, repr=False, default=None)
    name: str = "FedAvg"

    def __post_init__(self) -> None:
        """
        Validate hyperparameters.

        Raises:
            ValueError: if hyperparameters are invalid.

        """
        if self.step_size <= 0:
            raise ValueError("`step_size` must be positive")
        if self.num_local_epochs <= 0:
            raise ValueError("`num_local_epochs` must be positive")

    def initialize(self, network: FedNetwork) -> None:  # noqa: D102
        self.x0 = alg_helpers.zero_initialization(self.x0, network)
        self._setup_rngs(network.clients)
        network.server.initialize(x=self.x0, received_msgs=dict.fromkeys(network.clients, self.x0))
        for client in network.clients:
            client.initialize(x=self.x0, received_msgs={network.server: self.x0})

    def _setup_rngs(self, clients: Sequence["Agent"]) -> None:
        if self.sgd_seed is not None:
            self._sgd_rngs = {client.id: random.Random(self.sgd_seed + client.id) for client in clients}
        else:
            self._sgd_rngs = None

    def step(self, network: FedNetwork, iteration: int) -> None:  # noqa: D102
        selected_clients = self._selected_clients_for_round(network, iteration)
        if not selected_clients:
            return

        self._sync_server_to_clients(network, selected_clients)
        self._run_local_updates(network, selected_clients)
        self.aggregate(network, selected_clients)

    def _sync_server_to_clients(self, network: FedNetwork, selected_clients: Sequence["Agent"]) -> None:
        network.send(sender=network.server, receiver=selected_clients, msg=network.server.x)
        for client in selected_clients:
            network.receive(receiver=client, sender=network.server)

    def _run_local_updates(self, network: FedNetwork, selected_clients: Sequence["Agent"]) -> None:
        for client in selected_clients:
            client.x = self._compute_local_update(client, network.server)
            network.send(sender=client, receiver=network.server, msg=client.x)

    def _compute_local_update(self, client: "Agent", server: "Agent") -> "Array":
        local_x = iop.copy(client.messages[server])
        if isinstance(client.cost, EmpiricalRiskCost):
            cost = client.cost
            n_samples = cost.n_samples
            rng = self._client_rng(client)
            return self._epoch_minibatch_update(cost, local_x, cost.batch_size, n_samples, rng)

        for _ in range(self.num_local_epochs):
            grad = client.cost.gradient(local_x)
            local_x -= self.step_size * grad
        return local_x

    def _client_rng(self, client: "Agent") -> random.Random:
        if self._sgd_rngs is None:
            return random.Random()
        return self._sgd_rngs[client.id]

    def _epoch_minibatch_update(
        self,
        cost: EmpiricalRiskCost,
        local_x: "Array",
        per_client_batch: int,
        n_samples: int,
        rng: random.Random,
    ) -> "Array":
        for _ in range(self.num_local_epochs):
            indices = list(range(n_samples))
            rng.shuffle(indices)
            for start in range(0, n_samples, per_client_batch):
                batch_indices = indices[start : start + per_client_batch]
                grad = cost.gradient(local_x, indices=batch_indices)
                local_x -= self.step_size * grad
        return local_x


@dataclass(eq=False)
class DGD(P2PAlgorithm):
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

    iterations: int = 100
    step_size: float = 0.001
    x0: "Array | None" = None
    name: str = "DGD"

    def __post_init__(self) -> None:
        """
        Validate hyperparameters.

        Raises:
            ValueError: if hyperparameters are invalid.

        """
        if self.step_size <= 0:
            raise ValueError("`step_size` must be positive")

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
class ATC(P2PAlgorithm):
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

    iterations: int = 100
    step_size: float = 0.001
    x0: "Array | None" = None
    name: str = "ATC"

    def __post_init__(self) -> None:
        """
        Validate hyperparameters.

        Raises:
            ValueError: if hyperparameters are invalid.

        """
        if self.step_size <= 0:
            raise ValueError("`step_size` must be positive")

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
class SimpleGT(P2PAlgorithm):
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

    iterations: int = 100
    step_size: float = 0.001
    x0: "Array | None" = None
    name: str = "SimpleGT"

    def __post_init__(self) -> None:
        """
        Validate hyperparameters.

        Raises:
            ValueError: if hyperparameters are invalid.

        """
        if self.step_size <= 0:
            raise ValueError("`step_size` must be positive")

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
class ED(P2PAlgorithm):
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

    iterations: int = 100
    step_size: float = 0.001
    x0: "Array | None" = None
    name: str = "ED"

    def __post_init__(self) -> None:
        """
        Validate hyperparameters.

        Raises:
            ValueError: if hyperparameters are invalid.

        """
        if self.step_size <= 0:
            raise ValueError("`step_size` must be positive")

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
class AugDGM(P2PAlgorithm):
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

    iterations: int = 100
    step_size: float = 0.001
    x0: "Array | None" = None
    name: str = "Aug-DGM"

    def __post_init__(self) -> None:
        """
        Validate hyperparameters.

        Raises:
            ValueError: if hyperparameters are invalid.

        """
        if self.step_size <= 0:
            raise ValueError("`step_size` must be positive")

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
class WangElia(P2PAlgorithm):
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

    iterations: int = 100
    step_size: float = 0.001
    x0: "Array | None" = None
    name: str = "Wang-Elia"

    def __post_init__(self) -> None:
        """
        Validate hyperparameters.

        Raises:
            ValueError: if hyperparameters are invalid.

        """
        if self.step_size <= 0:
            raise ValueError("`step_size` must be positive")

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
class EXTRA(P2PAlgorithm):
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

    iterations: int = 100
    step_size: float = 0.001
    x0: "Array | None" = None
    name: str = "EXTRA"

    def __post_init__(self) -> None:
        """
        Validate hyperparameters.

        Raises:
            ValueError: if hyperparameters are invalid.

        """
        if self.step_size <= 0:
            raise ValueError("`step_size` must be positive")

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
class ATCTracking(P2PAlgorithm):
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

    iterations: int = 100
    step_size: float = 0.001
    x0: "Array | None" = None
    name: str = "ATC-Tracking"

    def __post_init__(self) -> None:
        """
        Validate hyperparameters.

        Raises:
            ValueError: if hyperparameters are invalid.

        """
        if self.step_size <= 0:
            raise ValueError("`step_size` must be positive")

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
class NIDS(P2PAlgorithm):
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

    iterations: int = 100
    step_size: float = 0.001
    x0: "Array | None" = None
    name: str = "NIDS"

    def __post_init__(self) -> None:
        """
        Validate hyperparameters.

        Raises:
            ValueError: if hyperparameters are invalid.

        """
        if self.step_size <= 0:
            raise ValueError("`step_size` must be positive")

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
class ADMM(P2PAlgorithm):
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

    iterations: int = 100
    rho: float = 1
    alpha: float = 0.5
    z0: "Array | None" = None
    name: str = "ADMM"

    def __post_init__(self) -> None:
        """
        Validate hyperparameters.

        Raises:
            ValueError: if hyperparameters are invalid.

        """
        if self.rho <= 0:
            raise ValueError("`rho` must be positive")
        if not (0 < self.alpha < 1):
            raise ValueError("`alpha` must be in (0, 1)")

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
class ATG(P2PAlgorithm):
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

    iterations: int = 100
    rho: float = 1
    alpha: float = 0.5
    gamma: float = 0.1
    delta: float = 0.001
    x0: "Array | None" = None
    name: str = "ATG"

    def __post_init__(self) -> None:
        """
        Validate hyperparameters.

        Raises:
            ValueError: if hyperparameters are invalid.

        """
        if self.rho <= 0:
            raise ValueError("`rho` must be positive")
        if not (0 < self.alpha < 1):
            raise ValueError("`alpha` must be in (0, 1)")
        if self.gamma <= 0:
            raise ValueError("`gamma` must be positive")
        if self.delta <= 0:
            raise ValueError("`delta` must be positive")

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
class DLM(P2PAlgorithm):
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

    iterations: int = 100
    step_size: float = 0.001
    penalty: float = 1
    x0: "Array | None" = None
    name: str = "DLM"

    def __post_init__(self) -> None:
        """
        Validate hyperparameters.

        Raises:
            ValueError: if hyperparameters are invalid.

        """
        if self.step_size <= 0:
            raise ValueError("`step_size` must be positive")
        if self.penalty <= 0:
            raise ValueError("`penalty` must be positive")

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
                s = iop.stack([i.x - x_j for x_j in i.messages.values()])
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
                s = iop.stack([i.x - x_j for x_j in i.messages.values()])
                i.aux_vars["s"] = iop.sum(s, dim=0)  # pyright: ignore[reportArgumentType]

            # step 3: update dual variable
            for i in network.active_agents(iteration):
                i.aux_vars["y"] += self.penalty * i.aux_vars["s"]


DecentralizedLinearizedADMM = DLM  # alias
