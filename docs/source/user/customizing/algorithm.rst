Algorithms
----------

Use built-in algorithms from :mod:`~decent_bench.distributed_algorithms`, or implement your own by subclassing
:class:`~decent_bench.distributed_algorithms.Algorithm`.

When implementing a custom algorithm, provide:

- ``initialize(network)``
- ``step(network, iteration)``
- optional ``finalize(network)``

Always update agent states through the network and interoperability abstractions so metrics and communication schemes
remain consistent.


Available algorithms
^^^^^^^^^^^^^^^^^^^^

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





.. _interoperability:

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


