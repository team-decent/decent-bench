.. _checkpointing:

Storing results and checkpointing
---------------------------------
As discussed above, defining a :class:`~decent_bench.utils.checkpoint_manager.CheckpointManager` instance and passing it
to the ``checkpoint_manager`` argument of :func:`~decent_bench.benchmark.benchmark`, :func:`~decent_bench.benchmark.compute_metrics`, :func:`~decent_bench.benchmark.display_metrics`
allows to store all the results.

In particular, these are stored in the folder specified at init of :class:`~decent_bench.utils.checkpoint_manager.CheckpointManager`
(folder ``results`` in the examples above). The folder, after a full execution of the benchmarking workflow, includes:

1. Becnchmark problem definition in ``{checkpoint_dir}/benchmark_problem.pkl``.
2. Progress checkpoints allowing benchmark resumption if interrupted; each algorithms gets a subfolder containing the checkpoints ``{checkpoint_dir}/algorithm_X``.
3. Metric computation results in ``{checkpoint_dir}/metric_computation.pkl``.
4. Plots in ``{checkpoint_dir}/results/plots_figX.png``.
5. Tables in ``{checkpoint_dir}/results/table.txt`` and ``{checkpoint_dir}/results/table.tex``.

The complete folder structure looks like this:

.. code-block:: text

    checkpoint_dir
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


Checkpointing options
^^^^^^^^^^^^^^^^^^^^^
The checkpointing behavior can be controlled via these parameters passed to the init of :class:`~decent_bench.utils.checkpoint_manager.CheckpointManager`:

- ``checkpoint_dir``: directory where the checkpoint manager should save results. The folder must be non-existent or empty at init.
- ``checkpoint_step``: the frequency with which checkpoints are stored; that is, a new checkpoint is stored every ``checkpoint_step`` iterations of an algorithm. If set to ``None``, the algorithms are checkpointed only at the end of the benchmark run.
- ``keep_n_checkpoints``: as checkpoints are potentially large, this option allows to specify how many of them should be stored at any given time. Defaults to 3.
- ``benchmark_metadata``: optional dictionary to store custom metadata about the benchmark run; this is saved in ``metadata.json``.

Combine a large ``checkpoint_step`` and small ``keep_n_checkpoints`` to reduce both the computational and storage load
of checkpoints.

The following example shows a benchmark run with a fully customized :class:`~decent_bench.utils.checkpoint_manager.CheckpointManager`.
   
.. literalinclude:: ../../../examples/checkpointing_fed_custom_options.py
    :language: python
    :linenos:


Resuming benchmarks
^^^^^^^^^^^^^^^^^^^
If the previous benchmark is interrupted at any time, using :func:`~decent_bench.benchmark.resume_benchmark`. This will
complete to run all the trials (``n_trials``) for the specified number of iterations (``iterations`` argument of algorithms).

.. code-block:: python

    from decent_bench import benchmark
    from decent_bench.utils.checkpoint_manager import CheckpointManager

    if __name__ == "__main__":

        cm = CheckpointManager(checkpoint_dir="results")  # the folder created by the interrupted benchmark run

        results = benchmark.resume_benchmark(
            checkpoint_manager=cm,
            create_backup=True,  # creates a backup zip before resuming
        )


If more iterations and/or trials are needed, these can be performed starting from the output of a previous benchmark run.

.. code-block:: python

    from decent_bench import benchmark
    from decent_bench.utils.checkpoint_manager import CheckpointManager

    if __name__ == "__main__":

        cm = CheckpointManager(checkpoint_dir="results")  # the folder created by the previous benchmark run

        results = benchmark.resume_benchmark(
            checkpoint_manager=cm,
            create_backup=True,       # creates a backup zip before resuming
            increase_iterations=150,  # run 150 additional iterations
            increase_trials=10,       # run 10 additional trials
        )


Computing and displaying metrics later
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
decent-bench allows to compute and display metrics at different times. This is especially useful when :func:`~decent_bench.benchmark.benchmark`
takes a long time: one can run the benchmark in one script, and then apply :func:`~decent_bench.benchmark.compute_metrics`
and :func:`~decent_bench.benchmark.display_metrics` in another script, or in two separate scripts.

Loading a previously computed :class:`~decent_bench.benchmark._benchmark_result.BenchmarkResult` for metrics
computation can be done by:

.. code-block:: python

    from decent_bench import benchmark
    from decent_bench.utils.checkpoint_manager import CheckpointManager

    if __name__ == "__main__":

        cm = CheckpointManager(checkpoint_dir="results")  # the folder created by the benchmark run

        metrics_result = benchmark.compute_metrics(
            benchmark_result=None,
            checkpoint_manager=cm
        )

where setting ``benchmark_result=None`` tells `compute_metrics` to load previously computed results. As discussed before,
the set of metrics to be computed can be customized via the ``table_metrics`` and ``plot_metrics`` arguments.

Loading a previously computed :class:`~decent_bench.benchmark.MetricResult` for display can be done similarly by:

.. code-block:: python

    from decent_bench import benchmark
    from decent_bench.utils.checkpoint_manager import CheckpointManager

    if __name__ == "__main__":

        cm = CheckpointManager(checkpoint_dir="results")  # the folder where compute_metrics stored its results

        benchmark.display_metrics(
            metrics_result=None,
            checkpoint_manager=cm,
        )

Additionally, the :class:`~decent_bench.benchmark.MetricResult` can be loaded beforehand for inspection, via its
``table_metrics``, ``plot_metrics``, ``algorithms`` properties, which return a list of available metrics/algorithms.

.. code-block:: python

    from decent_bench import benchmark
    from decent_bench.utils.checkpoint_manager import CheckpointManager

    if __name__ == "__main__":

        cm = CheckpointManager(checkpoint_dir="results")  # the folder where compute_metrics stored its results

        metrics_result = cm.load_metrics_result()

        print("Available algorithms:", metrics_result.algorithms)
        print("Available table metrics:", metrics_result.table_metrics)
        print("Available plot metrics:", metrics_result.plot_metrics)

        benchmark.display_metrics(
            metrics_result=metrics_result,
            checkpoint_manager=cm,
            table_metrics=["x error"],          # select only some metrics
            plot_metrics=["gradient norm"],     # or
            algorithms=["Scaffold"],            # algorithms
        )
