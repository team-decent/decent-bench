Benchmarking
============

TODO: This page covers the standard benchmark workflow and the most important settings.
:doc:`customizing <customizing>` will show how to customize each component of the benchmark

Running a benchmark
-------------------

A typical run has three phases:

1. Execute algorithms with :func:`~decent_bench.benchmark.benchmark`.
2. Compute metrics with :func:`~decent_bench.benchmark.compute_metrics`.
3. Display or save outputs with :func:`~decent_bench.benchmark.display_metrics`.

.. literalinclude:: ../../../test/user-guide/benchmarking_minimal.py
   :language: python

Customizing benchmark settings
------------------------------

The most commonly tuned settings are:

- ``n_trials`` and ``max_processes`` for statistical robustness and runtime
- ``progress_step`` and ``show_speed`` for execution feedback
- ``table_metrics`` and ``plot_metrics`` for output selection
- ``confidence_level`` for confidence intervals

.. literalinclude:: ../../../test/user-guide/benchmarking_custom_settings.py
   :language: python

Reproducibility (setting a seed)
--------------------------------

For reproducible experiments, set seeds consistently for all random sources you use.

- Python random module
- NumPy
- framework-specific RNGs (for example PyTorch)
- graph generation utilities that accept ``seed``

.. code-block:: python
    import random
    import numpy as np
    import networkx as nx
    random.seed(0)
    np.random.seed(0)
    graph = nx.random_regular_graph(d=3, n=100, seed=0)
If your benchmark uses framework-level randomness, also set that framework's seed at startup.
Use a fixed seed per experiment when comparing algorithms, and change the seed between experiment batches when you
want robustness checks.


Sunny case
----------
Benchmark algorithms on a regression problem without any communication constraints, using only default settings.

Generally benchmark execution involves three steps:

1. Run algorithms on a :class:`~decent_bench.benchmark.BenchmarkProblem` object and get results in a :class:`~decent_bench.benchmark.BenchmarkResult` object.
2. Compute metrics from the benchmark results, which returns a :class:`~decent_bench.benchmark.MetricResult` object.
3. Display the computed metrics in tables and plots.

Note:
    When running benchmarks, be sure to guard the execution code with ``if __name__ == "__main__":`` to avoid issues with multiprocessing on some platforms (e.g., Windows).
    This is a common Python practice to ensure that the benchmark code only runs when the script is executed directly, and not when it is imported as a module or when worker 
    processes are spawned for multiprocessing. If you forget to include this guard and you are using multiprocessing, i.e. with ``max_processes > 1`` in :func:`~decent_bench.benchmark.benchmark`, 
    you may encounter errors or unexpected behavior due to the way multiprocessing works on different platforms.

**The following is a working example. The remainder of the user guide will be updated soon.**




.. literalinclude:: ../../examples/basic_p2p_example.py
    :language: python
    :linenos:


.. literalinclude:: ../../examples/basic_p2p_example.py
    :language: python
    :linenos:
    :lineno-start: 12
    :lines: 12-28


.. code-block:: text

    algorithm      FedAvg               Scaffold                                                                                                                                                                                                                                             
    metric                                                                                                                                                                                                                                                                                              
    gradient norm  3.01e-01 ± 0.00e+00  1.95e-15 ± 0.00e+00                                                                                                                                                                                                                                             
    x error        2.46e-02 ± 0.00e+00  2.40e-16 ± 0.00e+00


.. code-block:: text

    algorithm      ADMM                 DGD                  ED                                                                                                                                                                                                              
    metric                                                                                                                                                                                                                                                                                 
    gradient norm  2.38e-15 ± 0.00e+00  4.30e-02 ± 0.00e+00  5.43e-11 ± 0.00e+00                                                                                                                                                                                                              
    x error        2.51e-16 ± 0.00e+00  4.25e-03 ± 0.00e+00  5.56e-12 ± 0.00e+00 


Benchmark executions will have outputs like these:

.. list-table::

   * - .. image:: ../_static/basic_p2p_example_plots.png
          :align: center
          :height: 350px
     - .. image:: ../_static/basic_fed_example_plots.png
          :align: center
          :height: 350px




Execution settings
------------------
Configure settings for metrics, trials, logging, and multiprocessing.

.. code-block:: python

    from logging import DEBUG
    import numpy as np

    from decent_bench.metrics import utils
    from decent_bench import benchmark
    from decent_bench.costs import LinearRegressionCost
    from decent_bench.distributed_algorithms import ADMM, DGD
    from decent_bench.metrics import ComputationalCost
    from decent_bench.metrics.metric_library import GradientCalls, Regret
    from decent_bench.metrics.runtime_library import RuntimeLoss, RuntimeRegret

    if __name__ == "__main__":
        regret = Regret(x_log=False, y_log=True)
        gradient_calls = GradientCalls([min, np.average, max, sum])

        benchmark_result = benchmark.benchmark(
            algorithms=[DGD(iterations=1000, step_size=0.01), ADMM(iterations=1000, penalty=10, relaxation=0.3)],
            benchmark_problem=benchmark.create_regression_problem(LinearRegressionCost),
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


Interpreting Metric Warnings
----------------------------
Warnings during metric computation and display are expected in some scenarios, especially when algorithms diverge or
when metric prerequisites are not provided.

Unavailable metrics
~~~~~~~~~~~~~~~~~~~
Some metrics require additional problem information.

- ``regret`` and ``x error`` require ``problem.x_optimal``.
- ``accuracy``, ``mse``, ``precision``, and ``recall`` require ``problem.test_data`` and
    agents with :class:`~decent_bench.costs.EmpiricalRiskCost`.
- ``accuracy``, ``precision``, and ``recall`` additionally require integer-valued targets.
- Federated server metrics such as ``server accuracy`` and ``server mse`` require
    :class:`~decent_bench.networks.FedNetwork` and use the explicit server model history.

If these requirements are not met, the metric marks itself unavailable, a warning is given, and is omitted from the final
metric lists returned by :func:`~decent_bench.benchmark.compute_metrics`.

Plot truncation warnings
~~~~~~~~~~~~~~~~~~~~~~~~
Plot metric trajectories are truncated at the first non-finite datapoint (NaN/inf) or first datapoint above the
internal plotting threshold used for log-scale stability.

Typical messages include:

- ``Truncating plot computation ... retained K point(s) from M/N trial(s).``
- ``Skipping plot computation ... all trials diverged before the first plottable datapoint.``

These warnings indicate that divergence was detected and handled gracefully for plotting.

Why tables can show NaN while plots still appear
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tables and plots summarize different parts of the trajectory:

- Table metrics typically use the final iteration.
- Plot metrics may retain and display an earlier finite prefix.

So it is expected to see ``nan ± nan`` in a table for a metric while still seeing a corresponding curve in the plot.








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

    from decent_bench import benchmark
    from decent_bench.costs import LinearRegressionCost
    from decent_bench.distributed_algorithms import ADMM, DGD
    from decent_bench.metrics.runtime_library import RuntimeLoss, RuntimeRegret
    from decent_bench.utils.checkpoint_manager import CheckpointManager

    if __name__ == "__main__":
        checkpoint_manager = CheckpointManager(checkpoint_dir="benchmark_results/my_experiment")

        # Saves step 1.
        benchmark_result = benchmark.benchmark(
            algorithms=[
                DGD(iterations=10000, step_size=0.001),
                ADMM(iterations=10000, penalty=10, relaxation=0.3),
            ],
            benchmark_problem=benchmark.create_regression_problem(LinearRegressionCost),
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

    from decent_bench import benchmark
    from decent_bench.costs import LinearRegressionCost
    from decent_bench.distributed_algorithms import DGD
    from decent_bench.utils.checkpoint_manager import CheckpointManager

    if __name__ == "__main__":
        benchmark.benchmark(
            algorithms=[DGD(iterations=50000, step_size=0.001)],
            benchmark_problem=benchmark.create_regression_problem(LinearRegressionCost),
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

    from decent_bench import benchmark
    from decent_bench.costs import LinearRegressionCost
    from decent_bench.distributed_algorithms import DGD
    from decent_bench.utils.checkpoint_manager import CheckpointManager
    import platform

    if __name__ == "__main__":
        checkpoint_manager = CheckpointManager(checkpoint_dir="benchmark_results/my_experiment")

        benchmark_result = benchmark.benchmark(
            algorithms=[DGD(iterations=1000, step_size=0.001)],
            benchmark_problem=benchmark.create_regression_problem(LinearRegressionCost),
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


Loading :class:`~decent_bench.benchmark.MetricResult` for displaying metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Similarly, you can load previously computed metrics by setting ``metrics_result`` to ``None`` in :func:`~decent_bench.benchmark.display_metrics` and providing the same checkpoint manager.
The loaded :class:`~decent_bench.benchmark.MetricResult` exposes ``algorithms``,
``table_metrics``, and ``plot_metrics`` to discover valid filter values.

.. code-block:: python

    from decent_bench import benchmark
    from decent_bench.utils.checkpoint_manager import CheckpointManager

    if __name__ == "__main__":
        checkpoint_manager = CheckpointManager(checkpoint_dir="benchmark_results/my_experiment")

        metrics_result = checkpoint_manager.load_metrics_result()
        if metrics_result is None:
            raise ValueError("No computed metrics found in checkpoint directory")

        print("Available algorithms:", metrics_result.algorithms)
        print("Available table metrics:", metrics_result.table_metrics)
        print("Available plot metrics:", metrics_result.plot_metrics)

        benchmark.display_metrics(
            metrics_result=metrics_result,
            checkpoint_manager=checkpoint_manager,
            algorithms=["DGD"],
            table_metrics=["nr gradient calls"],
            plot_metrics=["regret"],
            save_path=checkpoint_manager.get_results_path(),
        )
