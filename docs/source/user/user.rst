User Guide
==========


Available algorithms
--------------------

Note: algorithms are slightly modified with respect to the corresponding papers to ensure that they work in a broader
range of conditions (asynchorny/partial participation, loss of communications, etc) than the original papers.

Peer-to-peer
~~~~~~~~~~~~
P2P algorithms: :tagged:`peer-to-peer`


ADMM+P2P: :tagged:`peer-to-peer, ADMM`

metrics: :tagged:`metric`
runtime metrics: :tagged:`runtime metric`


Federated
~~~~~~~~~

federated algorithms: :tagged:`federated`


:class:`~decent_bench.algorithms.federated.FedAvg` performs local client
gradient steps from the broadcast server model and then replaces the server
model with the uniform average of the received final client models.

:class:`~decent_bench.algorithms.federated.FedProx` extends
:class:`~decent_bench.algorithms.federated.FedAvg` with a proximal coefficient
``penalty``. In the FedProx equations, the symbol :math:`\mu` is used for this
coefficient, and the corresponding argument is ``penalty``. Setting
``penalty=0`` reduces :class:`~decent_bench.algorithms.federated.FedProx`
to :class:`~decent_bench.algorithms.federated.FedAvg`.

:class:`~decent_bench.algorithms.federated.FedAdagrad`,
:class:`~decent_bench.algorithms.federated.FedYogi`, and
:class:`~decent_bench.algorithms.federated.FedAdam` form the built-in
:class:`~decent_bench.algorithms.federated.FedOpt` family. They keep the same
client-side local SGD structure as
:class:`~decent_bench.algorithms.federated.FedAvg`, but each client uploads its
model delta to the server and the server applies an adaptive optimizer update
instead of plain averaging to the next global iterate.

:class:`~decent_bench.algorithms.federated.FedNova` supports the plain
Local-SGD FedNova update together with optional local momentum, proximal local
updates, and server momentum. Clients accumulate their local updates together
with the FedNova normalization coefficient and upload them to the server in
two separate transmissions before the server applies the aggregated update.
If none of the first-phase normalizer uploads are received in a round, that round is skipped without updating the server model.
The ``num_local_steps`` argument can be either a single integer for
homogeneous local work or a per-client mapping for heterogeneous local step
counts.

:class:`~decent_bench.algorithms.federated.FedLT` implements Federated Local
Training with cost-driven local gradients. Its ``local_solver`` option supports
``"gd"``, ``"nesterov"``, and ``"adam"`` local updates. Solver-specific
hyperparameters are passed through ``solver_args``. If ``solver_args`` is omitted 
or left empty, solver-specific defaults are used: ``"nesterov"`` uses ``momentum=0.9``, 
``"adam"`` uses ``beta1=0.9``, ``beta2=0.999``, and ``epsilon=1e-8``, and ``"gd"`` uses no additional solver arguments.
:class:`~decent_bench.costs.EmpiricalRiskCost` objects use their default
mini-batch gradient behavior, so gradient-based local solvers use mini-batches,
while generic :class:`~decent_bench.costs.Cost` objects use full gradients.
Fed-LT compression and Fed-PLT-style noisy-message experiments are configured
through :class:`~decent_bench.networks.FedNetwork`
``message_compression`` and ``message_noise`` schemes rather than algorithm
arguments.

:class:`SCAFFOLD <decent_bench.algorithms.federated.Scaffold>` implements
stochastic controlled averaging with a server control variate and one client
control variate per client. Selected clients run local steps corrected by the
difference between the server and client control variates, then upload both
their model delta and control-variate delta. The server applies the averaged
model delta with ``server_step_size`` and updates its control variate using the
participation fraction.

:class:`~decent_bench.algorithms.federated.FedPD` implements the FedPD
primal-dual update. Clients keep persistent local primal variables, dual
variables, and centre candidates. Each round uses ``num_local_steps`` local
gradient steps on the augmented Lagrangian, with ``step_size`` controlling the
local learning rate and ``penalty`` controlling the FedPD quadratic
penalty/dual-update parameter. The ``skip_probability`` argument controls the
paper's optional communication skipping: ``0`` aggregates every round and ``1``
always keeps local centre candidates without server aggregation. Like
``FedNova``, ``num_local_steps`` can be either a single integer or a per-client
mapping. FedPD always uses all active clients; partial participation is not
supported.

:class:`~decent_bench.algorithms.federated.FedDyn` implements Federated Dynamic
Regularization. Clients keep a persistent dynamic state ``g`` and the server
keeps an auxiliary vector ``h``. Selected clients run ``num_local_steps`` local
gradient steps on the dynamic-regularized local objective using ``step_size`` as
the local learning rate and ``penalty`` as the regularization coefficient. In
the FedDyn equations, the symbol :math:`\alpha` denotes this regularization,
and the corresponding argument is ``penalty``.

Federated aggregation
^^^^^^^^^^^^^^^^^^^^^
For the built-in federated algorithms, aggregation affects only how client
updates are combined at the server. It does not change the optimization
objective itself. If you want to optimize a weighted objective, scale the
client costs in the problem definition instead of relying on aggregation.

:class:`~decent_bench.algorithms.federated.FedAvg` and
:class:`~decent_bench.algorithms.federated.FedProx` use uniform averaging
over the selected clients.

:class:`~decent_bench.algorithms.federated.FedAdagrad`,
:class:`~decent_bench.algorithms.federated.FedYogi`, and
:class:`~decent_bench.algorithms.federated.FedAdam` also average client
model deltas uniformly over the selected clients before applying their
server-side adaptive optimizer.

:class:`~decent_bench.algorithms.federated.FedNova` uses data-proportional
client weights over the received uploads in the round, and it does not average
final client models directly when local step counts differ. Instead, it
aggregates the client cumulative SGD updates using the FedNova rescaling factor
and applies the resulting descent step at the server. With the default flags,
this is the plain no-momentum FedNova variant. Enabling ``use_momentum``,
``use_prox``, or ``use_server_momentum`` extends the local and server updates
to the corresponding FedNova variants controlled by ``momentum``, ``penalty``,
and ``server_momentum``. In the FedNova equations, symbols :math:`\beta`,
:math:`\mu`, and :math:`\gamma` map to those arguments respectively. Plain
FedNova matches
:class:`~decent_bench.algorithms.federated.FedAvg` only when the participating
clients use the same number of local steps and
:class:`~decent_bench.algorithms.federated.FedNova` and
:class:`~decent_bench.algorithms.federated.FedAvg` both use data-proportional
aggregation weights.

:class:`~decent_bench.algorithms.federated.FedPD` aggregates received centre
candidates uniformly when communication is not skipped.

:class:`~decent_bench.algorithms.federated.FedDyn` averages the received
selected-client models uniformly, so the local-model average is scaled by the
number of received selected clients. The next server model also includes the
FedDyn dynamic correction from the server auxiliary vector ``h``. The ``h``
update uses only received client models and scales their model deltas by the
total number of clients.

:class:`~decent_bench.algorithms.federated.Scaffold` matches the standard
SCAFFOLD algorithm and always uses uniform averaging over the selected clients.

Client selection
^^^^^^^^^^^^^^^^^^^^^^^^^^
Federated algorithms accept a ``selection_scheme`` argument. The scheme receives
the active clients for the current round and returns the subset that should
receive the server broadcast.

:class:`~decent_bench.schemes.UniformSelection` samples active clients
uniformly without replacement. This is the default for the built-in federated
algorithms.

:class:`~decent_bench.schemes.DataSizeSelection` samples active clients
without replacement with probabilities proportional to client data size. This is
useful when local dataset sizes are imbalanced and you want larger clients to
have more participation opportunities. The scheme reads data size from
``EmpiricalRiskCost.n_samples``, so it requires clients to use empirical-risk
costs.

:class:`~decent_bench.schemes.FairSelection` prioritizes
clients with fewer past selections. This is useful when availability or random
sampling can repeatedly skip some clients and you want selection opportunities
to remain balanced across rounds.

:class:`~decent_bench.schemes.HighLossSelection` evaluates client losses
at each client's current ``x`` and selects the clients with highest loss.

.. code-block:: python

    from decent_bench.algorithms.federated import FedAvg
    from decent_bench.schemes import DataSizeSelection

    algorithm = FedAvg(
        iterations=100,
        step_size=0.01,
        selection_scheme=DataSizeSelection(fraction_selected_clients=0.1),
    )


Available costs
---------------

Regression
~~~~~~~~~~
:tagged:`cost, regression`

Classification
~~~~~~~~~~~~~~
:tagged:`cost, regression`


PyTorchCost regularization
~~~~~~~~~~~~~~~~~~~~~~~~~~
When combining :class:`~decent_bench.costs.PyTorchCost` with one of the
built-in regularizers, instantiate the regularizer with the same framework
and device as the empirical cost:

.. code-block:: python

    from decent_bench.costs import L2RegularizerCost
    from decent_bench.utils.types import SupportedFrameworks

    reg = L2RegularizerCost(
        shape=cost.shape,
        framework=SupportedFrameworks.PYTORCH,
        device=cost.device,
    )
    objective = cost + reg

This preserves compatibility with the PyTorch empirical objective and keeps
the resulting objective in the empirical, batch-compatible abstraction.
It is convenient for composition, but it is not necessarily the most
efficient option compared with native framework-specific regularization.







Benchmark problems
------------------

Configure out-of-the-box regression problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Configure communication constraints and other settings for out-of-the-box regression problems.

The ``agent_state_snapshot_period`` parameter controls how often metrics are recorded.
Setting it to a value greater than 1 (the default) can help reduce overhead for long-running algorithms while still providing enough data points for analysis.
This also speeds up metric calulation and plotting, which can be significant for large benchmarks with many iterations and agents.

.. code-block:: python

    from decent_bench import benchmark
    from decent_bench.costs import LinearRegressionCost
    from decent_bench.distributed_algorithms import ADMM, DGD, ED

    problem = benchmark.create_regression_problem(
        LinearRegressionCost,
        n_agents=100,
        agent_state_snapshot_period=10, # Record metrics every 10 iterations
        n_neighbors_per_agent=3,
        asynchrony=True,
        compression=True,
        noise=True,
        drops=True,
    )

    if __name__ == "__main__":
        benchmark_result = benchmark.benchmark(
            algorithms=[
                DGD(iterations=1000, step_size=0.01),
                ED(iterations=1000, step_size=0.01),
                ADMM(iterations=1000, penalty=10, relaxation=0.3),
            ],
            benchmark_problem=problem,
        )
        metrics_result = benchmark.compute_metrics(benchmark_result)
        benchmark.display_metrics(metrics_result)


Modify existing problems
~~~~~~~~~~~~~~~~~~~~~~~~
Change the settings of an already created benchmark problem, for example, the network topology.

.. code-block:: python

    import networkx as nx

    from decent_bench import benchmark
    from decent_bench.costs import LinearRegressionCost
    from decent_bench.distributed_algorithms import ADMM, DGD, ED

    n_agents = 100
    n_neighbors_per_agent = 3

    problem = benchmark.create_regression_problem(
        LinearRegressionCost,
        n_agents=n_agents,
        n_neighbors_per_agent=n_neighbors_per_agent,
        asynchrony=True,
        compression=True,
        noise=True,
        drops=True,
    )

    problem.network_structure = nx.random_regular_graph(n_agents, n_neighbors_per_agent)

    if __name__ == "__main__":
        benchmark_result = benchmark.benchmark(
            algorithms=[
                DGD(iterations=1000, step_size=0.01),
                ED(iterations=1000, step_size=0.01),
                ADMM(iterations=1000, penalty=10, relaxation=0.3),
            ],
            benchmark_problem=problem,
        )
        metrics_result = benchmark.compute_metrics(benchmark_result)
        benchmark.display_metrics(metrics_result)


Create problems using existing resources
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Create a custom benchmark problem using existing resources.

.. code-block:: python

    import networkx as nx

    from decent_bench import benchmark
    from decent_bench import centralized_algorithms as ca
    from decent_bench.benchmark import BenchmarkProblem
    from decent_bench.costs import LogisticRegressionCost
    from decent_bench.datasets import SyntheticClassificationData
    from decent_bench.distributed_algorithms import ADMM, DGD, ED
    from decent_bench.schemes import GaussianNoise, Quantization, UniformActivationRate, UniformDropRate
    from decent_bench.utils.types import SupportedFrameworks
    from decent_bench.utils.types import SupportedFrameworks

    n_agents = 100

    dataset = SyntheticClassificationData(
        n_classes=2, 
        n_partitions=n_agents, 
        n_samples_per_partition=10, 
        n_features=3, 
        framework=SupportedFrameworks.NUMPY,
    )

    costs = [LogisticRegressionCost(*p) for p in dataset.training_partitions()]

    sum_cost = sum(costs[1:], start=costs[0])
    x_optimal = ca.accelerated_gradient_descent(
        sum_cost, x0=None, max_iter=50000, stop_tol=1e-100, max_tol=1e-16
    )

    problem = BenchmarkProblem(
        network_structure=nx.random_regular_graph(3, n_agents, seed=0),
        costs=costs,
        x_optimal=x_optimal,
        agent_activations=[UniformActivationRate(0.5)] * n_agents,
        message_compression=Quantization(quantization_step=0.01),
        message_noise=GaussianNoise(mean=0, std=0.001),
        message_drop=UniformDropRate(drop_rate=0.5),
    )

    if __name__ == "__main__":
        benchmark_result = benchmark.benchmark(
            algorithms=[
                DGD(iterations=1000, step_size=0.01),
                ED(iterations=1000, step_size=0.01),
                ADMM(iterations=1000, penalty=10, relaxation=0.3),
            ],
            benchmark_problem=problem,
        )
        metrics_result = benchmark.compute_metrics(benchmark_result)
        benchmark.display_metrics(metrics_result)


Network constraints
^^^^^^^^^^^^^^^^^^^
Networks can model constrained communication through activation, compression,
noise, and message-drop schemes. These schemes are configured when creating a
network or benchmark problem and are applied automatically during
:meth:`~decent_bench.networks.Network.send`.

Activation schemes control whether an agent is available at a given iteration.
The built-in options are :class:`~decent_bench.schemes.AlwaysActive`,
:class:`~decent_bench.schemes.UniformActivationRate`,
:class:`~decent_bench.schemes.MarkovChainActivation`, and
:class:`~decent_bench.schemes.PoissonActivation`.
:class:`~decent_bench.schemes.CyclicActivation` alternates between active and
inactive intervals, with an optional phase offset to model staggered recurring
availability windows.
Implemented federated algorithms in decent-bench first ask the network for
active clients and then apply the client-selection scheme to that active set.

Compression schemes transform messages before they are sent. The default
:class:`~decent_bench.schemes.NoCompression` leaves messages unchanged.
:class:`~decent_bench.schemes.Quantization` rounds message entries to a fixed
number of significant digits. :class:`~decent_bench.schemes.TopK` and
:class:`~decent_bench.schemes.RandK` keep only a subset of coordinates and set
the rest to zero. :class:`~decent_bench.schemes.StochasticQuantization`
implements stochastic norm-scaled quantization used in QSGD.

Noise schemes perturb delivered messages. The default
:class:`~decent_bench.schemes.NoNoise` leaves messages unchanged, while
:class:`~decent_bench.schemes.GaussianNoise` adds independent Gaussian noise to
each message entry.

Drop schemes decide whether a message is lost before delivery. The default
:class:`~decent_bench.schemes.NoDrops` delivers every message.
:class:`~decent_bench.schemes.UniformDropRate` drops messages independently
with fixed probability, and :class:`~decent_bench.schemes.GilbertElliott`
models bursty drops with a two-state channel.

Message schemes can be provided as one shared instance or as a dictionary
mapping each sender agent to its own scheme. Use one instance per sender for
stateful schemes, since sharing a stateful scheme shares its internal state
across all senders.

.. code-block:: python

    from decent_bench.schemes import (
        GaussianNoise,
        StochasticQuantization,
        UniformActivationRate,
        UniformDropRate,
    )

    problem = BenchmarkProblem(
        network_structure=network_structure,
        costs=costs,
        x_optimal=x_optimal,
        agent_activations=[UniformActivationRate(0.8) for _ in costs],
        message_compression=StochasticQuantization(n_levels=8),
        message_noise=GaussianNoise(mean=0, std=0.001),
        message_drop=UniformDropRate(drop_rate=0.1),
    )

Create problems from scratch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Create a custom benchmark problem with your own dataset, cost function, and communication schemes by implementing the
corresponding abstracts.

.. code-block:: python

    import networkx as nx

    from decent_bench import benchmark
    from decent_bench import centralized_algorithms as ca
    from decent_bench.benchmark import BenchmarkProblem
    from decent_bench.costs import Cost
    from decent_bench.datasets import DatasetHandler
    from decent_bench.distributed_algorithms import DGD, SimpleGT
    from decent_bench.schemes import AgentActivationScheme, CompressionScheme, DropScheme, NoiseScheme

    class MyDataset(DatasetHandler): ... # Optional but convienient to manage partitions

    class MyCost(Cost): ...

    class MyAgentActivationScheme(AgentActivationScheme): ...

    class MyCompressionScheme(CompressionScheme): ...

    class MyNoiseScheme(NoiseScheme): ...

    class MyDropScheme(DropScheme): ...

    n_agents = 100

    costs = [MyCost(*p) for p in MyDataset().training_partitions()]

    sum_cost = sum(costs[1:], start=costs[0])
    x_optimal = ca.accelerated_gradient_descent(
        sum_cost, x0=None, max_iter=50000, stop_tol=1e-100, max_tol=1e-16
    )

    problem = BenchmarkProblem(
        network_structure=nx.random_regular_graph(3, n_agents, seed=0),
        costs=costs,
        x_optimal=x_optimal,
        agent_activations=[MyAgentActivationScheme()] * n_agents,
        message_compression=MyCompressionScheme(),
        message_noise=MyNoiseScheme(),
        message_drop=MyDropScheme(),
    )

    if __name__ == "__main__":
        benchmark_result = benchmark.benchmark(
            algorithms=[DGD(iterations=1000, step_size=0.01), SimpleGT(iterations=1000, step_size=0.01)],
            benchmark_problem=problem,
        )
        metrics_result = benchmark.compute_metrics(benchmark_result)
        benchmark.display_metrics(metrics_result)





Interoperability requirement
----------------------------
Decent-Bench is designed to interoperate with multiple array/tensor frameworks (NumPy, PyTorch, JAX, etc.). To keep
algorithms framework-agnostic, always use the interoperability layer :class:`~decent_bench.utils.interoperability`, aliased as
`iop`, and the :class:`~decent_bench.utils.array.Array` wrapper when creating, manipulating, and exchanging values:

- Use :class:`decent_bench.utils.interoperability.zeros` instead of framework-specific constructors (e.g., `np.zeros`, `torch.zeros`). 
    Other examples are :meth:`~decent_bench.utils.interoperability.ones_like`, :meth:`~decent_bench.utils.interoperability.uniform_like`, :meth:`~decent_bench.utils.interoperability.normal_like`, etc.
    See :mod:`~decent_bench.utils.interoperability` for a full list of available methods and :mod:`~decent_bench.algorithms` for examples of usage.
- Avoid calling any framework-specific functions directly within your algorithm. 
    Let the :class:`~decent_bench.costs.Cost` implementations handle framework-specific details for 
    :func:`~decent_bench.costs.Cost.function`, :func:`~decent_bench.costs.Cost.gradient`, :func:`~decent_bench.costs.Cost.hessian`, and :func:`~decent_bench.costs.Cost.proximal`.
- When you need to create a new array/tensor, use the interoperability layer to ensure compatibility with the agent's cost function framework and device.
    If a method to create your specific array/tensor is not available, see the implementation of :attr:`~decent_bench.networks.P2PNetwork.weights` as en example.


Philosophy
----------
To keep algorithm definitions consistent and easy to scan, we recommend using the following order for algorithm
dataclass fields:

1. ``iterations`` (required)
2. Hyperparameters (step size, penalty, number of local epochs, etc.)
3. Initialization parameters (e.g., ``x0``), with defaults
4. ``name``

This is a style guideline only; we do not enforce it programmatically.


Algorithms
----------
Create a new algorithm to benchmark against existing ones.

When implementing a custom algorithm by subclassing :class:`~decent_bench.algorithms.p2p.P2PAlgorithm`, you need to understand the following methods:

- **initialize(network)**: Called once before the algorithm starts. Use this to set up initial values for agents' primal variables (:attr:`Agent.x <decent_bench.agents.Agent.x>`), auxiliary variables (:attr:`Agent.aux_vars <decent_bench.agents.Agent.aux_vars>`), and received messages (:attr:`Agent.messages <decent_bench.agents.Agent.messages>`). **Implementation required.**
    If you want the agents' primal variable to be a customizable parameter to the algorithm, consider using a field like ``x0: Array | None = None`` in your algorithm class.
    Use a helper function like :func:`~decent_bench.algorithms.utils.initial_states` to initialize it properly if the input argument is ``None``. 
    :func:`~decent_bench.algorithms.utils.initial_states` initializes x0 to zero if x0 is None, otherwise uses provided x0. 
    :func:`~decent_bench.algorithms.utils.normal_initialization` can also be used to create normally distributed random initializations,
    and :func:`~decent_bench.algorithms.utils.uniform_initialization` for uniformly distributed;
    :func:`~decent_bench.algorithms.utils.pytorch_initialization` can be used with PyTorchCosts.

- **step(network, iteration)**: Called at each iteration of the algorithm. This is where the main algorithm logic goes - updating agent states, computing gradients, exchanging messages, etc. **Implementation required.**

- **finalize(network)**: Called once after all iterations complete. Use this for cleanup operations like clearing auxiliary variables to free memory. **Implementation optional** - the default implementation clears all auxiliary variables.

- **run(network)**: Orchestrates the full algorithm execution by calling :meth:`initialize <decent_bench.algorithms.Algorithm.initialize>`, then :meth:`step <decent_bench.algorithms.Algorithm.step>` for each iteration, and finally finalize. **You should NOT implement this** - it is already provided by the base :class:`~decent_bench.algorithms.Algorithm` class.

**Note**: In order for metrics to work, use :attr:`Agent.x <decent_bench.agents.Agent.x>` to update the local primal
variable **once** every iteration. If you need to perform multiple updates within an iteration, consider accumulating them and applying a single update at the end of the iteration. 
Similarly, in order for the benchmark problem's communication schemes to be applied, use the
:attr:`~decent_bench.networks.P2PNetwork`/ :attr:`~decent_bench.networks.FedNetwork` object to retrieve agents and to send and receive messages. 
Be sure to use :meth:`~decent_bench.networks.Network.active_agents` during algorithm runtime so that asynchrony is properly handled.
You can also inspect :attr:`~decent_bench.networks.Network.graph` to use NetworkX utilities (e.g., plotting or listing edges); mutating this graph changes the network topology.
In :class:`~decent_bench.networks.FedNetwork`, :meth:`~decent_bench.networks.Network.agents` and :meth:`~decent_bench.networks.Network.active_agents` refer to clients (the server is available via :attr:`~decent_bench.networks.FedNetwork.server`/ :attr:`~decent_bench.networks.FedNetwork.coordinator`).
Federated networks enforce an always-available server: a custom server passed to :class:`~decent_bench.networks.FedNetwork` must use :class:`~decent_bench.schemes.AlwaysActive`, otherwise network construction raises ``ValueError``.
The agents/clients lists are cached for efficiency, so the network graph should be treated as immutable after construction.
Federated aggregation affects only how client updates are combined at the server and does not change the objective
being optimized. If you want to optimize a weighted objective :math:`\min \sum_i w_i f_i(x)`, scale each local cost by
``w_i`` when defining the problem.

.. code-block:: python

    import decent_bench.utils.algorithm_helpers as alg_helpers
    import decent_bench.utils.interoperability as iop
    from decent_bench import benchmark
    from decent_bench.costs import LinearRegressionCost
    from decent_bench.distributed_algorithms import ADMM, DGD, P2PAlgorithm
    from decent_bench.networks import P2PNetwork
    from decent_bench.utils.array import Array

    class MyNewAlgorithm(P2PAlgorithm):
        iterations: int
        step_size: float
        x0: Array | None = None
        name: str = "MNA"

        # Initialize agents with Array values using the interoperability layer
        def initialize(self, network: P2PNetwork) -> None:  # noqa: D102
            self.x0 = alg_helpers.zero_initialization(self.x0, network)
            for agent in network.agents():
                y0 = iop.zeros(shape=agent.cost.shape, framework=agent.cost.framework, device=agent.cost.device)
                neighbors = network.neighbors(agent)
                agent.initialize(x=self.x0, received_msgs=dict.fromkeys(neighbors, self.x0), aux_vars={"y": y0})

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

        def finalize(self, network: P2PNetwork) -> None:  # noqa: D102
            # Optionally override finalize method. Code below is the default behavior
            # which clears auxiliary variables to free memory.
            # This function is called after the algorithm completes.
            # It is generally not necessary to override this method unless your algorithm
            # requires special cleanup or finalization.
            for agent in network.agents():
                agent.aux_vars.clear()

    if __name__ == "__main__":
        benchmark.benchmark(
            algorithms=[
                MyNewAlgorithm(iterations=1000, step_size=0.01),
                DGD(iterations=1000, step_size=0.01),
                ADMM(iterations=1000, penalty=10, relaxation=0.3),
            ],
            benchmark_problem=benchmark.create_regression_problem(LinearRegressionCost),
        )


Metrics
-------

Table and plot metrics
~~~~~~~~~~~~~~~~~~~~~~
Create your own metrics to tabulate and/or plot.

.. code-block:: python
    
    from collections.abc import Sequence

    import numpy.linalg as la
    import decent_bench.utils.interoperability as iop

    from decent_bench.metrics import utils
    from decent_bench import benchmark
    from decent_bench.agents import AgentMetricsView
    from decent_bench.benchmark import BenchmarkProblem
    from decent_bench.costs import LinearRegressionCost
    from decent_bench.distributed_algorithms import ADMM, DGD
    from decent_bench.metrics import Metric

    class XError(Metric):

        description: str = "x error"

        def compute(  # noqa: D102
            self,
            agents: Sequence[AgentMetricsView],
            problem: BenchmarkProblem,
            iteration: int,
        ) -> list[float]:
            if problem.x_optimal is None:
                return [float("nan") for _ in agents]

            x_optimal_np = iop.to_numpy(problem.x_optimal)

            if iteration == -1:
                return [float(la.norm(x_optimal_np - iop.to_numpy(a.x_history[a.x_history.max()]))) for a in agents]
            return [
                float(la.norm(x_optimal_np - iop.to_numpy(a.x_history[iteration])))
                for a in agents
            ]

    if __name__ == "__main__":
        x_error = XError(
            statistics=[min, max],
            fmt=".4e",
            x_log=False,
            y_log=True,
        )

        benchmark_result = benchmark.benchmark(
            algorithms=[
                DGD(iterations=1000, step_size=0.01),
                ADMM(iterations=1000, penalty=10, relaxation=0.3),
            ],
            benchmark_problem=benchmark.create_regression_problem(LinearRegressionCost),
        )

        metrics_result = benchmark.compute_metrics(benchmark_result, table_metrics=[x_error], plot_metrics=[x_error])
        benchmark.display_metrics(metrics_result)


Runtime metrics
~~~~~~~~~~~~~~~
Create your own runtime metrics to monitor algorithm progress during execution.

Runtime metrics are computed during algorithm execution to provide live feedback for early stopping or monitoring convergence.
Unlike post-hoc table and plot metrics, runtime metrics don't store historical data and are designed to be lightweight.
They are updated at a specified interval and can optionally save plots to disk after execution.

.. code-block:: python

    from collections.abc import Sequence

    import decent_bench.utils.interoperability as iop
    from decent_bench import benchmark
    from decent_bench.agents import Agent
    from decent_bench.benchmark import BenchmarkProblem
    from decent_bench.costs import LinearRegressionCost
    from decent_bench.distributed_algorithms import DGD
    from decent_bench.metrics import RuntimeMetric

    class RuntimeConsensusError(RuntimeMetric):
        """Monitors how well agents agree on their decision variables."""

        description = "Consensus Error"
        x_log = False
        y_log = True

        def compute(self, problem: BenchmarkProblem, agents: Sequence[Agent], iteration: int) -> float:
            # Compute average x across all agents
            x_avg = iop.mean(iop.stack([agent.x for agent in agents]), dim=0)
            
            # Compute average distance from the mean
            errors = [float(iop.norm(agent.x - x_avg)) for agent in agents]
            return sum(errors) / len(agents)

    class RuntimeRegret(RuntimeMetric):
        """Example how to cache computations"""

        description = "Regret"
        x_log = False
        y_log = False

        def compute(self, problem: BenchmarkProblem, agents: Sequence[Agent], iteration: int) -> float:
            if problem.x_optimal is None:
                return float("nan")

            agent_cost = sum(agent.cost.function(agent.x) for agent in agents) / len(agents)

            if hasattr(self, "_cached_optimal_cost"):
                return agent_cost - self._cached_optimal_cost

            # Since x_optimal is fixed for the problem, we can cache the optimal cost after computing it once
            self._cached_optimal_cost: float = sum(agent.cost.function(problem.x_optimal) for agent in agents) / len(agents)

            return agent_cost - self._cached_optimal_cost

    if __name__ == "__main__":
        benchmark_result = benchmark.benchmark(
            algorithms=[DGD(iterations=10000, step_size=0.001)],
            benchmark_problem=benchmark.create_regression_problem(LinearRegressionCost),
            runtime_metrics=[
                RuntimeConsensusError(
                    update_interval=100,  # Compute and plot every 100 iterations
                    save_path="results",  # Save plots to "results" directory during execution
                )
            ],
        )

**Important considerations for runtime metrics:**

- Keep the :meth:`~decent_bench.metrics.RuntimeMetric.compute` method efficient, as it's called during algorithm execution
- Avoid expensive computations that might significantly slow down the algorithm
- The ``update_interval`` parameter controls the trade-off between monitoring granularity and performance overhead
- If ``save_path`` is provided, plots are saved to disk at each update interval
- Runtime metrics are useful for early stopping, detecting divergence, or monitoring specific convergence properties


Cost Functions
--------------
Create new cost functions by subclassing :class:`~decent_bench.costs.Cost` and using interoperability decorators to keep
your implementation framework-agnostic. The decorators automatically wrap inputs/outputs as `Array` and ensure
compatibility with the selected framework and device of your custom cost.
Composition preserves specialized structure when possible, and otherwise falls back to generic wrappers.

Basic Operations
~~~~~~~~~~~~~~~~
Supported operations for cost objects:

- Addition: ``cost_a + cost_b``
- Subtraction: ``cost_a - cost_b``
- Negation: ``-cost``
- Scalar multiplication: ``scalar * cost`` or ``cost * scalar``
- Scalar division: ``cost / scalar``
- Summation: ``sum(costs)`` (uses ``__radd__``)

Composition Rules
~~~~~~~~~~~~~~~~~
Cost arithmetic preserves specialized structure for the most common composition patterns and falls back to generic
wrappers otherwise. When a composition falls back to :class:`~decent_bench.costs.SumCost` or
:class:`~decent_bench.costs.ScaledCost`, the result only guarantees the base :class:`~decent_bench.costs.Cost`
interface.

- ``regularizer_a + regularizer_b``, ``regularizer_a - regularizer_b``, ``scalar * regularizer``,
  ``regularizer / scalar``, and ``-regularizer`` preserve a regularizer-aware cost.
- ``scalar * empirical_cost`` preserves the empirical-risk interface through an internal empirical scaling wrapper.
- ``empirical_cost + regularizer`` and ``empirical_cost - regularizer`` preserve the empirical-risk interface through
  :class:`~decent_bench.costs.EmpiricalRegularizedCost`.
- Unsupported combinations still fall back to the generic wrappers
  :class:`~decent_bench.costs.SumCost` and :class:`~decent_bench.costs.ScaledCost`.

Regularization
~~~~~~~~~~~~~~
Regularized objectives can be built by composing cost functions with arithmetic. Decent-Bench provides the following
built-in regularizers:

The canonical regularization pattern is ``objective = cost + lambda_ * regularizer``.

- :class:`~decent_bench.costs.L1RegularizerCost` for :math:`\|x\|_1`
- :class:`~decent_bench.costs.L2RegularizerCost` for :math:`\frac{1}{2}\|x\|_2^2`
- :class:`~decent_bench.costs.FractionalQuadraticRegularizerCost` for
  :math:`\sum_i \frac{x_i^2}{1 + x_i^2}` (nonconvex)

All built-in regularizers accept and ignore empirical-risk-specific kwargs (for example ``indices="batch"``), so
batching continues to work when you compose them with empirical risk costs.

Empirical Risk Composition
~~~~~~~~~~~~~~~~~~~~~~~~~~
Supported empirical-risk compositions preserve empirical-risk-specific behavior such as ``predict``, ``dataset``,
``n_samples``, ``batch_size``, and batch helpers.

In particular, ``objective = cost + regularizer`` returns an
:class:`~decent_bench.costs.EmpiricalRegularizedCost`, which combines the empirical and regularizer contributions in
``function``, ``gradient``, and ``hessian`` while preserving the empirical interface of the base loss.

When using :class:`~decent_bench.costs.PyTorchCost`, prefer PyTorch's built-in loss regularizers for better
efficiency; iop regularizers remain available for cross-framework compatibility.

Examples
~~~~~~~~
.. code-block:: python

    from decent_bench.costs import (
        LogisticRegressionCost,
        L1RegularizerCost,
        L2RegularizerCost,
        FractionalQuadraticRegularizerCost,
    )

    cost = LogisticRegressionCost(dataset=dataset, batch_size="all")

    lam = 0.1
    eps = 0.01
    l1 = lam * L1RegularizerCost(shape=cost.shape)
    l2 = lam * L2RegularizerCost(shape=cost.shape)
    fq = eps * FractionalQuadraticRegularizerCost(shape=cost.shape)

    regularizer = l1 + l2
    objective = cost + regularizer
    nonconvex_objective = objective + fq

.. code-block:: python

    lambda_ = 0.05
    objective = cost + lambda_ * L2RegularizerCost(shape=cost.shape)
    value = objective.function(x, indices="all")
    gradient = objective.gradient(x, indices="all")

.. code-block:: python

    import numpy as np
    from decent_bench.costs import QuadraticCost

    arbitrary = QuadraticCost(np.eye(cost.shape[0]), np.zeros(cost.shape[0]))
    generic = objective + arbitrary  # returns SumCost

Important Semantics
~~~~~~~~~~~~~~~~~~~
Reduction Semantics
^^^^^^^^^^^^^^^^^^^
:class:`~decent_bench.costs.EmpiricalRegularizedCost.gradient` uses broadcast semantics when ``reduction=None``: the
empirical term returns one gradient per selected sample, and the regularizer gradient is added to each row. Averaging
over the sample dimension recovers the same composite gradient returned by ``reduction="mean"``.

Proximal Semantics
^^^^^^^^^^^^^^^^^^
.. warning::

    Proximal support is intentionally conservative. Positive scalar scaling preserves proximal support, and a single
    positively scaled regularizer term preserves the underlying regularizer proximal. Multi-term regularizer
    composites and :class:`~decent_bench.costs.EmpiricalRegularizedCost` do not define a generic proximal. Use a
    specialized proximal if one exists, or rely on :func:`decent_bench.centralized_algorithms.proximal_solver` when
    its assumptions are satisfied.

.. warning::

    :class:`~decent_bench.costs.SumCost.proximal` computes the proximal of the full summed objective through
    :func:`decent_bench.centralized_algorithms.proximal_solver`, which uses accelerated gradient descent. This
    requires the summed objective to satisfy that backend's assumptions, in particular differentiability, global
    smoothness, and convexity.

Copy Semantics
^^^^^^^^^^^^^^
.. warning::

    Composition wrappers keep references to the underlying cost objects; they do not make implicit copies. Mutating a
    reused cost after composition therefore affects all wrappers that reference it. Agent-installed call-counting
    hooks on reused cost objects are also shared. Use ``copy.deepcopy`` when independent composed objects or
    independent counting behavior are required.

.. code-block:: python

    import copy

    shared = cost + cost
    independent = copy.deepcopy(shared)

Custom Cost Example
~~~~~~~~~~~~~~~~~~~
.. code-block:: python

    from numpy import float64
    from numpy.typing import NDArray

    import decent_bench.utils.interoperability as iop
    from decent_bench.costs import Cost
    from decent_bench.utils.types import SupportedFrameworks, SupportedDevices

    class MyCost(Cost):
        def __init__(self, A: Array, b: Array):
            # Convert any external arrays to Array using the chosen framework/device
            self.A: NDArray[float64] = iop.to_numpy(A)
            self.b: NDArray[float64] = iop.to_numpy(b)

        @property
        def shape(self) -> tuple[int, ...]:
            # Domain shape (e.g., dimension of x)
            return (self.A.shape[1],)

        @property
        def framework(self) -> str:
            return SupportedFrameworks.NUMPY

        @property
        def device(self) -> str | None:
            return SupportedDevices.CPU

        @property
        def m_smooth(self) -> float:
            # Provide a meaningful smoothness constant if available
            return 1.0

        @property
        def m_cvx(self) -> float:
            # Provide convexity constant (0 if non-strongly convex)
            return 0.0

        @iop.autodecorate_cost_method(Cost.function)
        def function(self, x: NDArray[float64]) -> float:
            # Return a scalar (float) or Array scalar compatible with the framework
            r = self.A @ x - self.b
            return 0.5 * float(iop.dot(r, r))

        @iop.autodecorate_cost_method(Cost.gradient)
        def gradient(self, x: NDArray[float64]) -> NDArray[float64]:
            # Return an Array with same shape as x
            return self.A.T @ (self.A @ x - self.b)

        @iop.autodecorate_cost_method(Cost.hessian)
        def hessian(self, x: NDArray[float64]) -> NDArray[float64]:
            # Optional: return an Array representing the Hessian
            return self.A.T @ self.A

        @iop.autodecorate_cost_method(Cost.proximal)
        def proximal(self, x: NDArray[float64], rho: float) -> NDArray[float64]:
            # Optional: provide a closed-form proximal if available
            # Otherwise you can rely on `centralized_algorithms.proximal_solver`.
            return x  # identity as a placeholder

        # No __add__ implementation is required unless you want to preserve
        # a more specialized structure than the generic Cost fallback.
