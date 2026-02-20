User Guide
==========
This user guide shows you different examples of how to use decent-bench.


Installation
------------
Requires `Python 3.13+ <https://www.python.org/downloads/>`_

.. code-block:: bash

    pip install decent-bench


Sunny case
----------
Benchmark algorithms on a regression problem without any communication constraints, using only default settings.

Generally benchmark execution involves three steps:

1. Run algorithms on a :class:`~decent_bench.benchmark_problem.BenchmarkProblem` object and get results in a :class:`~decent_bench.benchmark.BenchmarkResult` object.
2. Compute metrics from the benchmark results, which returns a :class:`~decent_bench.benchmark.MetricsResult` object.
3. Display the computed metrics in tables and plots.

.. code-block:: python

    from decent_bench import benchmark, benchmark_problem
    from decent_bench.costs import LinearRegressionCost
    from decent_bench.distributed_algorithms import ADMM, DGD

    if __name__ == "__main__":
        # Run algorithms and get results in a BenchmarkResult object
        benchmark_result = benchmark.benchmark(
            algorithms=[
                DGD(iterations=1000, step_size=0.001),
                ADMM(iterations=1000, rho=10, alpha=0.3),
            ],
            benchmark_problem=benchmark_problem.create_regression_problem(LinearRegressionCost),
        )

        # Compute metrics from the benchmark results
        metrics_result = benchmark.compute_metrics(benchmark_result)

        # Display the computed metrics in tables and plots
        benchmark.display_metrics(metrics_result)


Benchmark executions will have outputs like these:

.. list-table::

   * - .. image:: _static/table.png
          :align: center
          :height: 350px
     - .. image:: _static/plot.png
          :align: center
          :height: 350px


Execution settings
------------------
Configure settings for metrics, trials, statistical confidence level, logging, and multiprocessing.

.. code-block:: python

    from logging import DEBUG
    import numpy as np

    import decent_bench.metrics.metric_utils as utils
    from decent_bench import benchmark, benchmark_problem
    from decent_bench.costs import LinearRegressionCost
    from decent_bench.distributed_algorithms import ADMM, DGD
    from decent_bench.metrics import ComputationalCost
    from decent_bench.metrics.metric_collection import GradientCalls, Regret
    from decent_bench.metrics.runtime_collection import RuntimeLoss, RuntimeRegret

    if __name__ == "__main__":
        regret = Regret([utils.single], x_log=False, y_log=True)
        gradient_calls = GradientCalls([min, np.average, max, sum])

        benchmark_result = benchmark.benchmark(
            algorithms=[DGD(iterations=1000, step_size=0.001), ADMM(iterations=1000, rho=10, alpha=0.3)],
            benchmark_problem=benchmark_problem.create_regression_problem(LinearRegressionCost),
            n_trials=10,
            max_processes=1,
            progress_step=200,
            show_speed=True,
            log_level=DEBUG,
            runtime_metrics=[
                RuntimeLoss(update_interval=100, save_path="results"), # Runtime plots are saved to "results" directory after execution
                RuntimeRegret(update_interval=100, save_path="results"),
            ],
        )

        metrics_result = benchmark.compute_metrics(
            benchmark_result, 
            table_metrics=[regret, gradient_calls],
            plot_metrics=[regret],
            confidence_level=0.9,
            log_level=DEBUG,
        )

        benchmark.display_metrics(
            metrics_result,
            table_metrics=[gradient_calls], # Can modify which metrics are displayed, cannot add new metrics that were not computed
            plot_metrics=[], # Disable plot metrics to only show tables
            table_fmt="latex",
            plot_grid=False,
            individual_plots=False,
            computational_cost=ComputationalCost(proximal=2.0, communication=0.1),
            compare_iterations_and_computational_cost=True,
            save_path="results", # Save tables and plots to the "results" directory
        )

For plot metrics some special options are available which change how they are displayed. 
If you set ``individual_plots = True`` then each plot metric will be displayed in its own separate figure,
otherwise groups of up to 3 plot metrics will be displayed together in the same figure as subplots.
If you set ``compare_iterations_and_computational_cost = True``, then an additional column of subplots will be added to each figure comparing iterations and computational cost, 
which can be useful to understand the trade-off between faster convergence and higher computational cost for different algorithms.
If you set ``plot_metrics`` to a list of lists of :class:`~decent_bench.metrics.Metric` objects (`list[list[Metric]]`), then each inner list will be plotted together in the same figure as subplots,
allowing you to control which metrics are plotted together.


Benchmark problems
------------------

Configure out-of-the-box regression problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Configure communication constraints and other settings for out-of-the-box regression problems.

The ``agent_state_snapshot_period`` parameter controls how often metrics are recorded.
Setting it to a value greater than 1 (the default) can help reduce overhead for long-running algorithms while still providing enough data points for analysis.
This also speeds up metric calulation and plotting, which can be significant for large benchmarks with many iterations and agents.

.. code-block:: python

    from decent_bench import benchmark, benchmark_problem
    from decent_bench.costs import LinearRegressionCost
    from decent_bench.distributed_algorithms import ADMM, DGD, ED

    problem = benchmark_problem.create_regression_problem(
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
                DGD(iterations=1000, step_size=0.001),
                ED(iterations=1000, step_size=0.001),
                ADMM(iterations=1000, rho=10, alpha=0.3),
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

    from decent_bench import benchmark, benchmark_problem
    from decent_bench.costs import LinearRegressionCost
    from decent_bench.distributed_algorithms import ADMM, DGD, ED

    n_agents = 100
    n_neighbors_per_agent = 3

    problem = benchmark_problem.create_regression_problem(
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
                DGD(iterations=1000, step_size=0.001),
                ED(iterations=1000, step_size=0.001),
                ADMM(iterations=1000, rho=10, alpha=0.3),
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
    from decent_bench.benchmark_problem import BenchmarkProblem
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
        message_compression=Quantization(n_significant_digits=4),
        message_noise=GaussianNoise(mean=0, sd=0.001),
        message_drop=UniformDropRate(drop_rate=0.5),
    )

    if __name__ == "__main__":
        benchmark_result = benchmark.benchmark(
            algorithms=[
                DGD(iterations=1000, step_size=0.001),
                ED(iterations=1000, step_size=0.001),
                ADMM(iterations=1000, rho=10, alpha=0.3),
            ],
            benchmark_problem=problem,
        )
        metrics_result = benchmark.compute_metrics(benchmark_result)
        benchmark.display_metrics(metrics_result)


Create problems from scratch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Create a custom benchmark problem with your own dataset, cost function, and communication schemes by implementing the
corresponding abstracts.

.. code-block:: python

    import networkx as nx

    from decent_bench import benchmark
    from decent_bench import centralized_algorithms as ca
    from decent_bench.benchmark_problem import BenchmarkProblem
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
            algorithms=[DGD(iterations=1000, step_size=0.001), SimpleGT(iterations=1000, step_size=0.001)],
            benchmark_problem=problem,
        )
        metrics_result = benchmark.compute_metrics(benchmark_result)
        benchmark.display_metrics(metrics_result)


Network utilities
-----------------
Plot a network explicitly when you need it:

.. code-block:: python

    import networkx as nx
    from decent_bench import benchmark_problem
    from decent_bench.utils import network_utils
    from decent_bench.costs import LinearRegressionCost

    problem = benchmark_problem.create_regression_problem(LinearRegressionCost, n_agents=25, n_neighbors_per_agent=3)

    # Plot using decent-bench helper (wraps :func:`networkx.drawing.nx_pylab.draw_networkx`)
    network_utils.plot_network(problem.network_structure, layout="circular", with_labels=True)

    # Or call NetworkX directly on the graph
    pos = nx.drawing.layout.spring_layout(problem.network_structure)
    nx.drawing.nx_pylab.draw_networkx(problem.network_structure, pos=pos, with_labels=True)

For more options, see the `NetworkX drawing guide <https://networkx.org/documentation/stable/reference/drawing.html>`_.


Storing results and checkpointing
----------------------------------
By default, benchmark progress and results (plots and tables) are only displayed but not saved to disk. To save results and enable
resumption of interrupted benchmarks, use the checkpoint functionality.

Basic checkpointing
~~~~~~~~~~~~~~~~~~~
Enable checkpointing by providing a :class:`~decent_bench.utils.checkpoint_manager.CheckpointManager` instance. This automatically saves:

1. Progress checkpoints allowing benchmark resumption if interrupted
2. Metric computation results to ``{checkpoint_dir}/metric_computation.pkl``
3. Plots to ``{checkpoint_dir}/results/plots_figX.png``
4. Tables to ``{checkpoint_dir}/results/table.txt`` and ``{checkpoint_dir}/results/table.tex``

where 3. and 4. are true if ``save_path`` is set to the checkpoint manager's results path in :func:`~decent_bench.benchmark.display_metrics`.

.. code-block:: python

    from decent_bench import benchmark, benchmark_problem
    from decent_bench.costs import LinearRegressionCost
    from decent_bench.distributed_algorithms import ADMM, DGD
    from decent_bench.metrics.runtime_collection import RuntimeLoss, RuntimeRegret
    from decent_bench.utils.checkpoint_manager import CheckpointManager

    if __name__ == "__main__":
        checkpoint_manager = CheckpointManager(checkpoint_dir="benchmark_results/my_experiment")

        # Saves step 1.
        benchmark_result = benchmark.benchmark(
            algorithms=[
                DGD(iterations=10000, step_size=0.001),
                ADMM(iterations=10000, rho=10, alpha=0.3),
            ],
            benchmark_problem=benchmark_problem.create_regression_problem(LinearRegressionCost),
            checkpoint_manager=checkpoint_manager,
            runtime_metrics=[
                RuntimeLoss(update_interval=100, save_path=checkpoint_manager.get_results_path()),
                RuntimeRegret(update_interval=100, save_path=checkpoint_manager.get_results_path()),
            ],
        )

        # Saves step 2.
        metrics_result = benchmark.compute_metrics(benchmark_result, checkpoint_manager)

        # Save step 3. and 4. if save_path is set to checkpoint_manager.get_results_path(), can be any other path as well.
        benchmark.display_metrics(
            metrics_result,
            save_path=checkpoint_manager.get_results_path(),
        )

The checkpoint directory structure:

.. code-block:: text

    benchmark_results/my_experiment/
    ├── metadata.json                   # Run configuration and algorithm metadata
    ├── benchmark_problem.pkl           # Initial benchmark problem state (before any trials)
    ├── initial_algorithms.pkl          # Initial algorithm states (before any trials)
    ├── initial_network.pkl             # Initial network state (before any trials)
    ├── metric_computation.pkl          # Computed metrics results (after all trials complete)
    ├── algorithm_0/                    # Directory for first algorithm
    │   ├── trial_0/                    # Directory for trial 0
    │   │   ├── checkpoint_0000100.pkl  # Combined algorithm+network state at iteration 100
    │   │   ├── checkpoint_0000200.pkl  # Combined algorithm+network state at iteration 200
    │   │   ├── progress.json           # {"last_completed_iteration": N}
    │   │   └── complete.json           # Marker file, contains path to final checkpoint
    │   ├── trial_1/
    │   │   └── ...
    │   └── trial_N/
    │       └── ...
    └── results/                        # Results directory for storing final tables and plots after completion
        ├── plots_fig1.png              # Final plot for figure 1 with plot results
        ├── plots_fig2.png              # Final plot for figure 2 with plot results
        ├── table.tex                   # Final LaTeX file with table results
        └── table.txt                   # Final text file with table results


Checkpoint parameters
~~~~~~~~~~~~~~~~~~~~~
Control checkpoint behavior with these parameters:

- **checkpoint_dir**: Directory path to save checkpoints. Must be empty or non-existent when starting a new benchmark.
- **checkpoint_step**: Number of iterations between checkpoints within each trial. If ``None``, only saves at trial completion. For long-running algorithms, use a value like 1000 to checkpoint during execution.
- **keep_n_checkpoints**: Maximum number of iteration checkpoints to keep per trial. Older checkpoints are automatically deleted to save disk space. Default is 3.
- **benchmark_metadata**: Optional dictionary to store custom metadata about the benchmark run (e.g., descriptions, system info, notes). This is saved in ``metadata.json`` and can be used for tracking and analysis.

.. code-block:: python

    import platform

    from decent_bench import benchmark, benchmark_problem
    from decent_bench.costs import LinearRegressionCost
    from decent_bench.distributed_algorithms import DGD
    from decent_bench.utils.checkpoint_manager import CheckpointManager

    if __name__ == "__main__":
        benchmark.benchmark(
            algorithms=[DGD(iterations=50000, step_size=0.001)],
            benchmark_problem=benchmark_problem.create_regression_problem(LinearRegressionCost),
            checkpoint_manager=CheckpointManager(
                checkpoint_dir="benchmark_results/long_run",
                checkpoint_step=5000,      # Checkpoint every 5000 iterations
                keep_n_checkpoints=5,      # Keep 5 most recent checkpoints
                benchmark_metadata={
                    "description": "Testing DGD step size sensitivity",
                    "system": platform.system(),
                    "python_version": platform.python_version(),
                    "notes": "Baseline run for paper experiments",
                },
            ),
        )


Resuming benchmarks
~~~~~~~~~~~~~~~~~~~
If a benchmark is interrupted, resume from the most recent checkpoint:

.. code-block:: python

    from decent_bench import benchmark
    from decent_bench.utils.checkpoint_manager import CheckpointManager

    if __name__ == "__main__":
        benchmark_result = benchmark.resume_benchmark(
            checkpoint_dir=CheckpointManager(checkpoint_dir="benchmark_results/my_experiment"),
            create_backup=True,  # Creates a backup zip before resuming
        )

Extend an existing benchmark with more iterations or trials:

.. code-block:: python

    from decent_bench import benchmark
    from decent_bench.utils.checkpoint_manager import CheckpointManager

    if __name__ == "__main__":
        benchmark_result = benchmark.resume_benchmark(
            checkpoint_dir=CheckpointManager(checkpoint_dir="benchmark_results/my_experiment"),
            increase_iterations=5000,  # Run 5000 additional iterations
            increase_trials=10,        # Run 10 additional trials
            create_backup=True,
        )

The optional parameters ``checkpoint_step`` and ``keep_n_checkpoints`` in :class:`~decent_bench.utils.checkpoint_manager.CheckpointManager` 
can be changed when resuming to control how frequently checkpoints are saved and how many are kept, allowing you to manage disk space for long-running benchmarks.


Saving metric computations
~~~~~~~~~~~~~~~~~~~~~~~~~~
By default, computed metrics are not saved unless a :class:`~decent_bench.utils.checkpoint_manager.CheckpointManager` is provided.
If provided, computed metrics are saved to ``{checkpoint_dir}/metric_computation.pkl`` after all trials complete. 
This allows you to preserve the results of expensive metric computations for later analysis without needing to recompute them.
This is useful when you want to modify plot settings, table formatting or :class:`~decent_bench.metrics.ComputationalCost` values after the benchmark has completed, without needing to rerun the entire benchmark or metric computation.

.. code-block:: python

    from decent_bench import benchmark, benchmark_problem
    from decent_bench.costs import LinearRegressionCost
    from decent_bench.distributed_algorithms import DGD
    from decent_bench.utils.checkpoint_manager import CheckpointManager
    import platform

    if __name__ == "__main__":
        checkpoint_manager = CheckpointManager(checkpoint_dir="benchmark_results/my_experiment")

        benchmark_result = benchmark.benchmark(
            algorithms=[DGD(iterations=1000, step_size=0.001)],
            benchmark_problem=benchmark_problem.create_regression_problem(LinearRegressionCost),
            checkpoint_manager=checkpoint_manager,
        )

        metrics_result = benchmark.compute_metrics(benchmark_result, checkpoint_manager)


Loading :class:`~decent_bench.benchmark._benchmark_result.BenchmarkResult` for metric computations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Furthermore, the checkpoint manager can be used to load previous benchmark results by setting ``benchmark_result`` to ``None`` in :func:`~decent_bench.benchmark.compute_metrics`,
making sure that the checkpoint manager is pointing to at least a partially completed benchmark. 
This allows you to compute new metrics from previously completed benchmarks or to modify existing metrics without needing to rerun the entire benchmark.
The new metrics will be saved to the checkpoint directory as described above.

.. code-block:: python

    from decent_bench import benchmark
    from decent_bench.utils.checkpoint_manager import CheckpointManager

    if __name__ == "__main__":
        checkpoint_manager = CheckpointManager(checkpoint_dir="benchmark_results/my_experiment")

        metrics_result = benchmark.compute_metrics(
            benchmark_result=None,
            checkpoint_manager=checkpoint_manager
        )


Loading :class:`~decent_bench.benchmark._metrics_result.MetricsResult` for displaying metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Similarly, you can load previously computed metrics by setting ``metrics_result`` to ``None`` in :func:`~decent_bench.benchmark.display_metrics` and providing the same checkpoint manager.

.. code-block:: python

    from decent_bench import benchmark
    from decent_bench.utils.checkpoint_manager import CheckpointManager

    if __name__ == "__main__":
        checkpoint_manager = CheckpointManager(checkpoint_dir="benchmark_results/my_experiment")

        benchmark.display_metrics(
            metrics_result=None, 
            checkpoint_manager=checkpoint_manager,
            save_path=checkpoint_manager.get_results_path(),
        )


Interoperability requirement
----------------------------
Decent-Bench is designed to interoperate with multiple array/tensor frameworks (NumPy, PyTorch, JAX, etc.). To keep
algorithms framework-agnostic, always use the interoperability layer :class:`~decent_bench.utils.interoperability`, aliased as
`iop`, and the :class:`~decent_bench.utils.array.Array` wrapper when creating, manipulating, and exchanging values:

- Use :class:`decent_bench.utils.interoperability.zeros` instead of framework-specific constructors (e.g., `np.zeros`, `torch.zeros`). 
    Other examples are :meth:`~decent_bench.utils.interoperability.ones_like`, :meth:`~decent_bench.utils.interoperability.rand_like`, :meth:`~decent_bench.utils.interoperability.randn_like`, etc.
    See :mod:`~decent_bench.utils.interoperability` for a full list of available methods and :mod:`~decent_bench.distributed_algorithms` for examples of usage.
- Avoid calling any framework-specific functions directly within your algorithm. 
    Let the :class:`~decent_bench.costs.Cost` implementations handle framework-specific details for 
    :func:`~decent_bench.costs.Cost.function`, :func:`~decent_bench.costs.Cost.gradient`, :func:`~decent_bench.costs.Cost.hessian`, and :func:`~decent_bench.costs.Cost.proximal`.
- When you need to create a new array/tensor, use the interoperability layer to ensure compatibility with the agent's cost function framework and device.
    If a method to create your specific array/tensor is not available, see the implementation of :attr:`~decent_bench.networks.P2PNetwork.weights` as en example.


Algorithms
----------
Create a new algorithm to benchmark against existing ones.

When implementing a custom algorithm by subclassing :class:`~decent_bench.distributed_algorithms.Algorithm`, you need to understand the following methods:

- **initialize(network)**: Called once before the algorithm starts. Use this to set up initial values for agents' primal variables (:attr:`Agent.x <decent_bench.agents.Agent.x>`), auxiliary variables (:attr:`Agent.aux_vars <decent_bench.agents.Agent.aux_vars>`), and received messages (:attr:`Agent.messages <decent_bench.agents.Agent.messages>`). **Implementation required.**
    If you want the agents' primal variable to be a customizable parameter to the algorithm, consider using a field like ``x0: Array | None = None`` in your algorithm class.
    Use a helper function like :func:`~decent_bench.utils.algorithm_helpers.zero_initialization` to initialize it properly if the input argument is ``None``. 
    :func:`~decent_bench.utils.algorithm_helpers.zero_initialization` initializes x0 to zero if x0 is None, otherwise uses provided x0. 
    :func:`~decent_bench.utils.algorithm_helpers.randn_initialization` can also be used to create normally distributed random initializations.

- **step(network, iteration)**: Called at each iteration of the algorithm. This is where the main algorithm logic goes - updating agent states, computing gradients, exchanging messages, etc. **Implementation required.**

- **finalize(network)**: Called once after all iterations complete. Use this for cleanup operations like clearing auxiliary variables to free memory. **Implementation optional** - the default implementation clears all auxiliary variables.

- **run(network)**: Orchestrates the full algorithm execution by calling :meth:`initialize <decent_bench.distributed_algorithms.Algorithm.initialize>`, then :meth:`step <decent_bench.distributed_algorithms.Algorithm.step>` for each iteration, and finally :meth:`finalize <decent_bench.distributed_algorithms.Algorithm.finalize>`. **You should NOT implement this** - it is already provided by the base :class:`~decent_bench.distributed_algorithms.Algorithm` class.

**Note**: In order for metrics to work, use :attr:`Agent.x <decent_bench.agents.Agent.x>` to update the local primal
variable **once** every iteration. If you need to perform multiple updates within an iteration, consider accumulating them and applying a single update at the end of the iteration. 
Similarly, in order for the benchmark problem's communication schemes to be applied, use the
:attr:`~decent_bench.networks.P2PNetwork`/ :attr:`~decent_bench.networks.FedNetwork` object to retrieve agents and to send and receive messages. 
Be sure to use :meth:`~decent_bench.networks.Network.active_agents` during algorithm runtime so that asynchrony is properly handled.
You can also inspect :attr:`~decent_bench.networks.Network.graph` to use NetworkX utilities (e.g., plotting or listing edges); mutating this graph changes the network topology.
In :class:`~decent_bench.networks.FedNetwork`, :meth:`~decent_bench.networks.Network.agents` and :meth:`~decent_bench.networks.Network.active_agents` refer to clients (the server is available via :attr:`~decent_bench.networks.FedNetwork.server`/ :attr:`~decent_bench.networks.FedNetwork.coordinator`).

.. code-block:: python

    import decent_bench.utils.algorithm_helpers as alg_helpers
    import decent_bench.utils.interoperability as iop
    from decent_bench import benchmark, benchmark_problem
    from decent_bench.costs import LinearRegressionCost
    from decent_bench.distributed_algorithms import ADMM, DGD, Algorithm
    from decent_bench.networks import P2PNetwork
    from decent_bench.utils.array import Array

    class MyNewAlgorithm(Algorithm):
        step_size: float
        x0: Array | None = None
        iterations: int = 100
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
                MyNewAlgorithm(iterations=1000, step_size=0.001),
                DGD(iterations=1000, step_size=0.001),
                ADMM(iterations=1000, rho=10, alpha=0.3),
            ],
            benchmark_problem=benchmark_problem.create_regression_problem(LinearRegressionCost),
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

    import decent_bench.metrics.metric_utils as utils
    from decent_bench import benchmark, benchmark_problem
    from decent_bench.agents import AgentMetricsView
    from decent_bench.benchmark_problem import BenchmarkProblem
    from decent_bench.costs import LinearRegressionCost
    from decent_bench.distributed_algorithms import ADMM, DGD
    from decent_bench.metrics import Metric

    class XError(Metric):

        table_description: str = "x error"
        plot_description: str = "x error"

        def get_data_from_trial(  # noqa: D102
            self,
            agents: Sequence[AgentMetricsView],
            problem: BenchmarkProblem,
            iteration: int,
        ) -> list[float]:
            if problem.x_optimal is None:
                return [float("nan") for _ in agents]

            x_optimal_np = iop.to_numpy(problem.x_optimal)

            if iteration == -1:
                return [float(la.norm(x_optimal_np - iop.to_numpy(a.x_history[max(a.x_history)]))) for a in agents]
            return [
                float(la.norm(x_optimal_np - iop.to_numpy(a.x_history[utils.find_closest_iteration(a, iteration)])))
                for a in agents
            ]

    if __name__ == "__main__":
        x_error = XError(
            statistics=[min, max],
            fmt=".4e",
            x_log=False,
            y_log=True,
            common_iterations=True, # Only plot at iterations common to all agents
        )

        benchmark_result = benchmark.benchmark(
            algorithms=[
                DGD(iterations=1000, step_size=0.001),
                ADMM(iterations=1000, rho=10, alpha=0.3),
            ],
            benchmark_problem=benchmark_problem.create_regression_problem(LinearRegressionCost),
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
    from decent_bench import benchmark, benchmark_problem
    from decent_bench.agents import Agent
    from decent_bench.benchmark_problem import BenchmarkProblem
    from decent_bench.costs import LinearRegressionCost
    from decent_bench.distributed_algorithms import DGD
    from decent_bench.metrics import RuntimeMetric

    class RuntimeConsensusError(RuntimeMetric):
        """Monitors how well agents agree on their decision variables."""

        description = "Consensus Error"

        def compute(self, problem: BenchmarkProblem, agents: Sequence[Agent], iteration: int) -> float:
            # Compute average x across all agents
            x_avg = iop.mean(iop.stack([agent.x for agent in agents]), dim=0)
            
            # Compute average distance from the mean
            errors = [float(iop.norm(agent.x - x_avg)) for agent in agents]
            return sum(errors) / len(agents)

    if __name__ == "__main__":
        benchmark_result = benchmark.benchmark(
            algorithms=[DGD(iterations=10000, step_size=0.001)],
            benchmark_problem=benchmark_problem.create_regression_problem(LinearRegressionCost),
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

Supported operations for cost objects:

- Addition: ``cost_a + cost_b``
- Subtraction: ``cost_a - cost_b``
- Negation: ``-cost``
- Scalar multiplication: ``scalar * cost`` or ``cost * scalar``
- Scalar division: ``cost / scalar``
- Summation: ``sum(costs)`` (uses ``__radd__``)

.. code-block:: python

    from numpy import float64
    from numpy.typing import NDArray

    import decent_bench.utils.interoperability as iop
    from decent_bench.costs import Cost, SumCost
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

        def __add__(self, other: Cost) -> Cost:
            # Support addition of costs
            if self.shape != other.shape:
                raise ValueError(f"Mismatching domain shapes: {self.shape} vs {other.shape}")

            return SumCost([self, other])
