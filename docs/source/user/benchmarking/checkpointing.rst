.. _checkpointing:

Storing results and checkpointing
---------------------------------
Pass ``storage_dir`` to :func:`~decent_bench.benchmark.configure` to save an experiment. decent-bench stores the
experiment metadata, benchmark problem, initial algorithm states, progress checkpoints, computed metrics, and displayed
tables and plots in that directory. When starting a new experiment, the directory must be empty or not yet exist.

The files and subdirectories have the following roles:

1. ``metadata.json`` records the backend, seed, checkpoint settings, number of trials, and algorithm information.
2. ``benchmark_problem.pkl.zst`` stores the initial problem state, while ``initial_algorithms.pkl.zst`` stores the
   algorithms before execution.
3. Each algorithm has a directory named ``algorithm_X``. Within it, each trial has a ``trial_Y`` directory containing
   progress information and compressed state snapshots. These snapshots let the run continue from its latest saved
   state if it is interrupted.
4. ``metric_computation.pkl.zst`` stores computed metrics after metrics computation.
5. The nested ``results/`` directory contains tables and plots created by ``display_metrics``.

The folder structure looks like this:

.. code-block:: text

    results/
    ├── metadata.json
    ├── benchmark_problem.pkl.zst
    ├── initial_algorithms.pkl.zst
    ├── algorithm_0/
    │   └── trial_0/
    │       ├── checkpoint_0000100.pkl.zst  # saved algorithm and network state
    │       ├── progress.json
    │       └── complete.json              # present when the trial completes
    ├── metric_computation.pkl.zst
    └── results/
        ├── plots_fig1.png
        ├── table.tex
        └── table.txt


Checkpointing options
^^^^^^^^^^^^^^^^^^^^^
Set checkpointing options in :func:`~decent_bench.benchmark.configure` when starting a new experiment:

* ``storage_dir``: directory for checkpoints and results.
* ``n_checkpoints``: number of checkpoints stored per trial, spaced across the run. The final iteration is always
  checkpointed. Defaults to 3; set to 1 to checkpoint only the final iteration.
* ``compression_level``: Zstandard compression level for checkpoint files. Defaults to 1.

These settings are stored in the experiment metadata and reused when reopening it. The following example configures
custom checkpoint settings:

.. literalinclude:: ../../../examples/checkpointing_fed_custom_options.py
    :language: python
    :linenos:


Resuming benchmarks
^^^^^^^^^^^^^^^^^^^
To continue an interrupted run, open the experiment by passing only its storage directory to ``configure``. The backend,
seed, and checkpointing options are read from ``metadata.json``. ``resume_benchmark`` completes pending trials. Use
``create_backup=True`` to save a zip backup before resuming.

.. code-block:: python

    from decent_bench.benchmark import configure, resume_benchmark

    if __name__ == "__main__":
        configure(storage_dir="results")
        results = resume_benchmark(create_backup=True)

To extend a completed run with additional iterations or trials, pass the increments to ``resume_benchmark``:

.. code-block:: python

    from decent_bench.benchmark import configure, resume_benchmark

    if __name__ == "__main__":
        configure(storage_dir="results")
        results = resume_benchmark(
            create_backup=True,
            increase_iterations=150,
            increase_trials=10,
        )

``increase_iterations`` adds that many iterations to each algorithm. ``increase_trials`` adds trials for each algorithm.
The existing results are retained, and the new work is appended to the experiment. A backup is recommended before
resuming a completed run.


Computing and displaying metrics later
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The benchmark, metrics computation, and display can run in separate processes. Configure each process with the same
``storage_dir``; when reopening an experiment, do not provide framework or device values.

Compute metrics from the stored benchmark result. Leaving ``benchmark_result`` unspecified tells
``compute_metrics`` to load the result from the configured experiment. USe the ``table_metrics`` and
``plot_metrics`` arguments to select which metrics to compute:

.. code-block:: python

    from decent_bench.benchmark import configure, compute_metrics

    configure(storage_dir="results")
    metrics_result = compute_metrics(
        table_metrics=[...],
        plot_metrics=[...],
    )

The returned :class:`~decent_bench.benchmark.MetricResult` can be inspected in this process, or saved automatically in
the experiment directory. In a separate process, call ``configure`` with the same path and leave ``metrics_result``
unspecified to load and display the saved metrics:

.. code-block:: python

    from decent_bench.benchmark import configure, display_metrics

    configure(storage_dir="results")
    display_metrics()

You can filter the displayed metrics and algorithms:

.. code-block:: python

    display_metrics(
        table_metrics=["x error"],
        plot_metrics=["gradient norm"],
        algorithms=["Scaffold"],
    )
