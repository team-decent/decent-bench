Benchmarking
============

This page covers the standard benchmark workflow and the most important settings. :doc:`This page <customizing>`
shows how to customize each component of the benchmark (from problem to algorithms to results).


Running a benchmark
-------------------
A typical benchmark run is characterized by four steps:

1. **Benchmark problem definition**: where the local costs :math:`f_i` (see :eq:`decentralized-problem`) and the network architecture are defined. This includes defining practical constraints such as limited communications/computational power. The benchmark problem is defined as a :class:`~decent_bench.benchmark.BenchmarkProblem` object.
2. **Benchmark**: where a set of algorithms is tested on the benchmark problem; see :func:`~decent_bench.benchmark.benchmark`. The results are contained in a :class:`~decent_bench.benchmark.BenchmarkResult` object.
3. **Compute metrics**: where selected performance metrics are computed based on the benchmark results; see :func:`~decent_bench.benchmark.compute_metrics`. The computed metrics are contained in a :class:`~decent_bench.benchmark.MetricResult` object.
4. **Display metrics**: where the metrics computed in step 3. are displayed as both tables and figures; see :func:`~decent_bench.benchmark.display_metrics`.

The worflow is depicted in the diagram below:

.. mermaid::

   flowchart TB
       START(( )):::empty -->|BenchmarkProblem| A[benchmark]
       A -->|BenchmarkResult| B[compute_metrics]
       B -->|MetricResult| C[display_metrics]

       classDef empty width:0px,height:0px,fill:transparent,stroke:transparent,color:transparent;


The following code examples show how to execute this workflow in practice. The first example is for a federated
setting, the second for a peer-to-peer setting, and after each example we show the corresponding output.


Federated example
^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../examples/basic_fed_example.py
    :language: python
    :linenos:

With table results:

.. code-block:: text

    algorithm                   FedAvg             Scaffold                                                                                                                                                                                                                                    
    metric                                                                                                                                                                                                                                                                                     
    gradient norm  3.71e-01 ± 0.00e+00  2.20e-14 ± 0.00e+00                                                                                                                                                                                                                                    
    x error        3.16e-01 ± 0.00e+00  1.42e-14 ± 0.00e+00  

and plots:

.. image:: ../_static/basic_fed_example_plots.png
    :align: center


Peer-to-peer example
^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../examples/basic_p2p_example.py
    :language: python
    :linenos:

With table results:

.. code-block:: text

    algorithm                     ADMM                  DGD                   ED                                                                                                                                                                                                               
    metric                                                                                                                                                                                                                                                                                     
    gradient norm  3.62e-14 ± 0.00e+00  1.60e-01 ± 0.00e+00  1.65e-08 ± 0.00e+00                                                                                                                                                                                                               
    x error        2.84e-14 ± 0.00e+00  1.90e-01 ± 0.00e+00  1.97e-08 ± 0.00e+00 

and plots:

.. image:: ../_static/basic_p2p_example_plots.png
    :align: center


Explanation
^^^^^^^^^^^
In both examples, we generate a linear regression problem

.. literalinclude:: ../../examples/basic_p2p_example.py
    :language: python
    :lines: 15

characterized by the local costs
:math:`f_i(x_i) = \frac{1}{m_i} \sum_{h = 1}^{m_i} \frac{1}{2} (a_i^h x_i - b_i^h)^2` with feature vectors
:math:`a_i^h \in \mathbb{R}^{1 \times n}` and targets :math:`b_i^h \in \mathbb{R}`. Since the costs are quadratic, the
``create_regression_problem`` utility also computes the optimal solution ``x_optimal``.

We then create the federated or peer-to-peer network of agents to solve the problem, respectively:

.. literalinclude:: ../../examples/basic_fed_example.py
    :language: python
    :lines: 14

.. literalinclude:: ../../examples/basic_p2p_example.py
    :language: python
    :lines: 17-18

where each agent is assigned one of the cost functions :math:`f_i` contained in ``costs``.

These steps set up the benchmark problem, which is represented by the data structure:

.. literalinclude:: ../../examples/basic_fed_example.py
    :language: python
    :lines: 15

The next steps are the execution of the benchmark using the :func:`~decent_bench.benchmark.benchmark` function, passing
a list of the algorithms to be tested as ``algorithms``, each with its hyperparameters. The results are contained in the
:class:`~decent_bench.benchmark.BenchmarkResult` object (``results`` in the examples), which can be used to compute the
performance metrics.

In the examples, two performance metrics are selected (which are instances of :class:`~decent_bench.metrics.Metric`):

.. literalinclude:: ../../examples/basic_fed_example.py
    :language: python
    :lines: 32

with :class:`~decent_bench.metrics.metric_library.XError` being the distance from the optimal solution, and
:class:`~decent_bench.metrics.metric_library.GradientNorm` the norm of the gradient of :math:`\sum_{i = 1}^N f_i`. If
the ``table_metrics`` and ``plot_metrics`` arguments of :func:`~decent_bench.benchmark.compute_metrics` are not specified,
all the available metrics are used instead. Not all metrics are always available: for example
:class:`~decent_bench.metrics.metric_library.XError` is not available if ``x_optimal`` is not available.

Note:
    In these example scripts, execution is guarded with ``if __name__ == "__main__":`` to avoid potential issues with
    multi-threading and some of the dependencies. Errors and unexpected results might appear if this guard is not used.

The following section explains in more detail how to customize the benchmark workflow.


Customizing benchmark settings
------------------------------

Storing results
^^^^^^^^^^^^^^^
A first important tool for benchmarking is the :class:`~decent_bench.utils.checkpoint_manager.CheckpointManager`, which
is instantiated in a folder location (folder ``results`` in the example below).

.. literalinclude:: ../../examples/checkpointing_fed_example.py
    :language: python
    :linenos:

The role of the :class:`~decent_bench.utils.checkpoint_manager.CheckpointManager` is to store the results at every step
of the benchmarking workflow. This is done by passing the  :class:`~decent_bench.utils.checkpoint_manager.CheckpointManager`
instance as the ``checkpoint_manager`` argument of :func:`~decent_bench.benchmark.benchmark`, :func:`~decent_bench.benchmark.compute_metrics`, :func:`~decent_bench.benchmark.display_metrics`.
Additionally, the :class:`~decent_bench.utils.checkpoint_manager.CheckpointManager` stores snapshots of the simulation
results ("checkpoints") as the benchmark runs. This allows resuming simulations later (*e.g.* adding more iterations)
or recovering interrupted simulations.

See more in :ref:`this section <checkpointing>`.


Benchmark settings
^^^^^^^^^^^^^^^^^^
The arguments of :func:`~decent_bench.benchmark.benchmark` allow customization of the benchmark run; the most important
arguments are:

* ``algorithms``: defines which algorithms should be tested. Each algorithm is an :class:`~decent_bench.algorithms.Algorithm` object initialized with the number of iterations to run (``iterations``) and required hyperparameters.
* ``benchmark_problem``: instance of :class:`~decent_bench.benchmark.BenchmarkProblem` which defines the problem.
* ``n_trials``: if the benchmark setup (network and/or algorithms) have stochastic features, running several trials is necessary. This is possible by setting the ``n_trials`` argument. See more in :ref:`this section <reproducibility>`.
* ``max_processes``: allows to set the number of threads that should be used to run the simulations.
* ``runtime_metrics``: if the benchmark run is very long, it can be useful to monitor its progress (beyond the progress bar that is displayed by default). This can be done by plotting :class:`~decent_bench.metrics.RuntimeMetric`, which they are simple and computationally cheap performance metric displayed in a plot that evolves as the simulations run. The available runtime metrics are: :tagged:`runtime metric`.


Metrics computation and display
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
As discussed above, the metrics are first computed with :func:`~decent_bench.benchmark.compute_metrics`, and then
displyed with :func:`~decent_bench.benchmark.display_metrics`.

.. literalinclude:: ../../examples/basic_p2p_example.py
    :language: python
    :lines: 36-44

These steps can be customized in several ways. For :func:`~decent_bench.benchmark.compute_metrics`:

* ``table_metrics`` and ``plot_metrics``: these can be different, since some metrics cannot be plotted (*e.g.* :class:`~decent_bench.metrics.metric_library.GradientCalls`, which counts the number of total gradient calls at the end of the simulation). Passing ``None`` to both arguments will use all the available metrics (already divided by table and plot metrics); see :mod:`~decent_bench.metrics.metric_library` for the list of available metrics. These arguments can also be an empty list to avoid computing either table or plot metrics.
* ``statistics_across_agents``: as discussed later in :ref:`this section <reproducibility>`, the benchmark can run several trials and average over the results. Additionally, several metrics yield one value for each agent (this is the case of :class:`~decent_bench.metrics.metric_library.GradientCalls`), and aggregation over the per-agent metrics is required. This can be controlled via the ``statistics_across_agents`` argument, which accepts a list of statistics to be computed across agents (options are: "mean", "std", "max", "min", "median"; "mean", "std" are used by default).
* Output: the function returns a :class:`~decent_bench.benchmark.MetricResult` object, which contains four ``pandas.DataFrame`` with the computed metrics (the raw data and the aggregated data across trials and agents). The results can thus be easily inspected.

For :func:`~decent_bench.benchmark.display_metrics`:

* ``table_metrics``, ``plot_metrics``, ``algorithms``: these can be used to select only a subset of the metrics/algorithms computed by :func:`~decent_bench.benchmark.compute_metrics` and stored in the :class:`~decent_bench.benchmark.MetricResult` object.
* Table formatting: ``table_fmt`` to choose either plain text or LaTeX tables (if a :class:`~decent_bench.utils.checkpoint_manager.CheckpointManager` is defined, both are stored in the results dir); * ``scale_compute`` to scale metrics related to the computational cost like :class:`~decent_bench.metrics.metric_library.GradientCalls`, which might be significantly large.
* Plot customization 1: ``plot_grid``; ``individual_plots`` to plot each metric in a separate figure; ``plot_format``.
* Plot customization 2: by default, metrics are plotted against iteration numbers; however, this might give a biased perspective since different algorithms will have different computational costs. The plots therefore can be customized to account for this by passing a :class:`~decent_bench.metrics.ComputationalCost` object into ``computational_cost``, which defines the cost of each operation (function, gradient, hessian, proximal evaluation, and communication). The computational cost will then replace the iteration numbers on the x-axis, or both can be plotted side-by-side using ``compare_iterations_and_computational_cost``. Finally, using computational cost for the x-axis might result in large values, and ``scale_x_axis`` can be used to make them more manageable.

The following example, with corresponding output, shows the above customization options in use.

.. literalinclude:: ../../examples/display_metrics_customization.py
    :language: python
    :linenos:

1) display with default options
"""""""""""""""""""""""""""""""
Plots:

.. list-table::
   :widths: 1 1

   * - .. figure:: ../_static/display_metrics_customization-1-2.png
          :align: center

     - .. figure:: ../_static/display_metrics_customization-1-1.png
          :align: center

Table:

.. code-block:: text

    algorithm                                         FedAvg              Scaffold                                                                                                                                                                                                             
    metric                    statistic                                                                                                                                                                                                                                                        
    client drift from server  mean       1.29e+01 ± 0.00e+00   0.00e+00 ± 0.00e+00                                                                                                                                                                                                             
                              std        7.07e+00 ± 0.00e+00   0.00e+00 ± 0.00e+00                                                                                                                                                                                                             
    consensus error           mean       1.29e+01 ± 0.00e+00   1.42e-14 ± 0.00e+00                                                                                                                                                                                                             
                              std        7.07e+00 ± 0.00e+00   0.00e+00 ± 0.00e+00                                                                                                                                                                                                             
    fraction selected clients                100.00% ± 0.00%       100.00% ± 0.00%                                                                                                                                                                                                             
    gradient norm                        5.96e-01 ± 0.00e+00   4.76e-14 ± 0.00e+00                                                                                                                                                                                                             
    loss                      mean       4.62e+01 ± 0.00e+00   2.73e+02 ± 0.00e+00                                                                                                                                                                                                             
                              std        4.80e+01 ± 0.00e+00   2.35e+02 ± 0.00e+00                                                                                                                                                                                                             
    nr Hessian calls          mean       0.00e+00 ± 0.00e+00   0.00e+00 ± 0.00e+00                                                                                                                                                                                                             
                              std        0.00e+00 ± 0.00e+00   0.00e+00 ± 0.00e+00                                                                                                                                                                                                             
    nr function calls         mean       0.00e+00 ± 0.00e+00   0.00e+00 ± 0.00e+00                                                                                                                                                                                                             
                              std        0.00e+00 ± 0.00e+00   0.00e+00 ± 0.00e+00                                                                                                                                                                                                             
    nr gradient calls         mean       2.50e+04 ± 0.00e+00   2.50e+04 ± 0.00e+00                                                                                                                                                                                                             
                              std        0.00e+00 ± 0.00e+00   0.00e+00 ± 0.00e+00                                                                                                                                                                                                             
    nr proximal calls         mean       0.00e+00 ± 0.00e+00   0.00e+00 ± 0.00e+00                                                                                                                                                                                                             
                              std        0.00e+00 ± 0.00e+00   0.00e+00 ± 0.00e+00                                                                                                                                                                                                             
    nr received messages      mean       4.55e+02 ± 0.00e+00   9.09e+02 ± 0.00e+00                                                                                                                                                                                                             
                              std        6.47e+02 ± 0.00e+00   1.29e+03 ± 0.00e+00                                                                                                                                                                                                             
    nr sent messages          mean       4.55e+02 ± 0.00e+00   9.09e+02 ± 0.00e+00                                                                                                                                                                                                             
                              std        6.47e+02 ± 0.00e+00   1.29e+03 ± 0.00e+00                                                                                                                                                                                                             
    nr sent messages dropped  mean       0.00e+00 ± 0.00e+00   0.00e+00 ± 0.00e+00                                                                                                                                                                                                             
                              std        0.00e+00 ± 0.00e+00   0.00e+00 ± 0.00e+00                                                                                                                                                                                                             
    nr x updates              mean       2.50e+02 ± 0.00e+00   2.50e+02 ± 0.00e+00                                                                                                                                                                                                             
                              std        0.00e+00 ± 0.00e+00   0.00e+00 ± 0.00e+00                                                                                                                                                                                                             
    regret                               1.74e-01 ± 0.00e+00  -1.36e-13 ± 0.00e+00                                                                                                                                                                                                             
    x error                              5.84e-01 ± 0.00e+00   5.68e-14 ± 0.00e+00 


2) display subset of plots, with iteration and computational cost side-by-side
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. image:: ../_static/display_metrics_customization-2.png
    :align: center


3) display only table in LaTeX format
"""""""""""""""""""""""""""""""""""""

.. code-block:: latex

    \begin{tabular}{llcc}                                                                                                                                                                                                                                                                      
    \toprule                                                                                                                                                                                                                                                                                   
    & algorithm & FedAvg & Scaffold \\                                                                                                                                                                                                                                                        
    metric & statistic &  &  \\                                                                                                                                                                                                                                                                
    \midrule                                                                                                                                                                                                                                                                                   
    \multirow[t]{2}{*}{client drift from server} & mean & 1.77e+01 ± 0.00e+00 & 0.00e+00 ± 0.00e+00 \\                                                                                                                                                                                         
    & std & 1.20e+01 ± 0.00e+00 & 0.00e+00 ± 0.00e+00 \\                                                                                                                                                                                                                                      
    \cline{1-4}                                                                                                                                                                                                                                                                                
    \multirow[t]{2}{*}{consensus error} & mean & 1.77e+01 ± 0.00e+00 & 0.00e+00 ± 0.00e+00 \\                                                                                                                                                                                                  
    & std & 1.20e+01 ± 0.00e+00 & 0.00e+00 ± 0.00e+00 \\                                                                                                                                                                                                                                      
    \cline{1-4}                                                                                                                                                                                                                                                                                
    fraction selected clients &  & 100.00% ± 0.00% & 100.00% ± 0.00% \\                                                                                                                                                                                                                        
    \cline{1-4}                                                                                                                                                                                                                                                                                
    gradient norm &  & 3.22e-01 ± 0.00e+00 & 1.42e-15 ± 0.00e+00 \\                                                                                                                                                                                                                            
    \cline{1-4}                                                                                                                                                                                                                                                                                
    \multirow[t]{2}{*}{loss} & mean & 4.10e+01 ± 0.00e+00 & 5.41e+02 ± 0.00e+00 \\                                                                                                                                                                                                             
    & std & 4.76e+01 ± 0.00e+00 & 5.25e+02 ± 0.00e+00 \\                                                                                                                                                                                                                                      
    \cline{1-4}                                                                                                                                                                                                                                                                                
    \multirow[t]{2}{*}{nr Hessian calls} & mean & 0.00e+00 ± 0.00e+00 & 0.00e+00 ± 0.00e+00 \\                                                                                                                                                                                                 
    & std & 0.00e+00 ± 0.00e+00 & 0.00e+00 ± 0.00e+00 \\                                                                                                                                                                                                                                      
    \cline{1-4}                                                                                                                                                                                                                                                                                
    \multirow[t]{2}{*}{nr function calls} & mean & 0.00e+00 ± 0.00e+00 & 0.00e+00 ± 0.00e+00 \\                                                                                                                                                                                                
    & std & 0.00e+00 ± 0.00e+00 & 0.00e+00 ± 0.00e+00 \\                                                                                                                                                                                                                                      
    \cline{1-4}                                                                                                                                                                                                                                                                                
    \multirow[t]{2}{*}{nr gradient calls} & mean & 2.50e+04 ± 0.00e+00 & 2.50e+04 ± 0.00e+00 \\                                                                                                                                                                                                
    & std & 0.00e+00 ± 0.00e+00 & 0.00e+00 ± 0.00e+00 \\                                                                                                                                                                                                                                      
    \cline{1-4}                                                                                                                                                                                                                                                                                
    \multirow[t]{2}{*}{nr proximal calls} & mean & 0.00e+00 ± 0.00e+00 & 0.00e+00 ± 0.00e+00 \\                                                                                                                                                                                                
    & std & 0.00e+00 ± 0.00e+00 & 0.00e+00 ± 0.00e+00 \\                                                                                                                                                                                                                                      
    \cline{1-4}                                                                                                                                                                                                                                                                                
    \multirow[t]{2}{*}{nr received messages} & mean & 4.55e+02 ± 0.00e+00 & 9.09e+02 ± 0.00e+00 \\                                                                                                                                                                                             
    & std & 6.47e+02 ± 0.00e+00 & 1.29e+03 ± 0.00e+00 \\                                                                                                                                                                                                                                      
    \cline{1-4}                                                                                                                                                                                                                                                                                
    \multirow[t]{2}{*}{nr sent messages} & mean & 4.55e+02 ± 0.00e+00 & 9.09e+02 ± 0.00e+00 \\                                                                                                                                                                                                 
    & std & 6.47e+02 ± 0.00e+00 & 1.29e+03 ± 0.00e+00 \\                                                                                                                                                                                                                                      
    \cline{1-4}                                                                                                                                                                                                                                                                                
    \multirow[t]{2}{*}{nr sent messages dropped} & mean & 0.00e+00 ± 0.00e+00 & 0.00e+00 ± 0.00e+00 \\                                                                                                                                                                                         
    & std & 0.00e+00 ± 0.00e+00 & 0.00e+00 ± 0.00e+00 \\                                                                                                                                                                                                                                      
    \cline{1-4}                                                                                                                                                                                                                                                                                
    \multirow[t]{2}{*}{nr x updates} & mean & 2.50e+02 ± 0.00e+00 & 2.50e+02 ± 0.00e+00 \\                                                                                                                                                                                                     
    & std & 0.00e+00 ± 0.00e+00 & 0.00e+00 ± 0.00e+00 \\                                                                                                                                                                                                                                      
    \cline{1-4}                                                                                                                                                                                                                                                                                
    regret &  & 4.15e-02 ± 0.00e+00 & 0.00e+00 ± 0.00e+00 \\                                                                                                                                                                                                                                   
    \cline{1-4}                                                                                                                                                                                                                                                                                
    x error &  & 2.58e-01 ± 0.00e+00 & 0.00e+00 ± 0.00e+00 \\                                                                                                                                                                                                                                  
    \cline{1-4}                                                                                                                                                                                                                                                                                
    \bottomrule                                                                                                                                                                                                                                                                                
    \end{tabular}


Interpreting logger messages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Throughout the benchmark workflow, logger messages are displayed in the terminal. The amount of information can be
tuned by setting the ``log_level`` argument of :func:`~decent_bench.benchmark.benchmark`, :func:`~decent_bench.benchmark.compute_metrics`, :func:`~decent_bench.benchmark.display_metrics`.
Examples are (printing progressively less information): ``logging.DEBUG``, ``logging.INFO`` (the default), ``logging.WARNING``, ``logging.ERROR``, ``logging.CRITICAL``.
See `here <https://docs.python.org/3/library/logging.html#logging-levels>`_ for more details.

The following are examples of the logger messages printed when running the code shown in the previous section with
the default ``log_level = logging.INFO``.

During benchmark problem creation:

.. code-block:: text

    INFO     Creating cost functions ...
    INFO     ... done!
    INFO     Finding the optimal solution to the problem ...
    INFO     ... done!                                              << with a progress bar if x_optimal is computed iteratively rather than in closed form
    INFO     Initialized checkpoint directory at 'results'          << if checkpoint_manager is defined

During benchmark run:

.. code-block:: text

    INFO     Starting benchmark execution
    Algorithm Progress Bar                                  Time   
    FedAvg    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
    Scaffold  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
    INFO     Benchmark execution complete

During metrics computation:

.. code-block:: text
    
    INFO     Starting metrics computation
    WARNING  Skipping table metric 'mse' because it is unavailable: requires problem.test_data
    ...                                                                                             << plus warnings for all other unavailable metrics
    Computing plot metrics   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00 Plot computation complete
    Computing table metrics  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00 Table computation complete
    INFO     Saved computed metrics result to results/metric_computation.pkl.zst                    << if checkpoint_manager is defined

During metrics display:

.. code-block:: text

    INFO     Displaying metrics
    WARNING  No available plot metrics were selected, skipping plots            << if plot_metrics = []
    INFO                                                                        << displayed table
    INFO     Compute counters (FunctionCalls, GradientCalls, HessianCalls, ProximalCalls) can yield very large numbers. Set ``scale_compute < 1`` to scale their values for display.
    WARNING  Metric 'consensus error' has y_log=True but contains non-positive y values. They were replaced with 1e-15 for plotting purposes. Non-positive values that were replaced (when close to 0, they are likely rounding errors): [0.0].
    INFO     Infinite/NaN values likely indicate algorithm divergence. Inspect plots to confirm.
    INFO     Saved LaTeX table to results/results/table.tex
    INFO     Saved text table to results/results/table.txt

where:

* The warning about 'consensus error' indicates that this metric should be plotted with a logarithmic y-axis, but it has a zero/negative value (in this case, zero); for plotting only, these values are replaced by :math:`10^{-15}` or a similarly small value.
* If any of the algorithms diverges (*e.g.* the selected step-size is too large), then inf or nan values will be displayed in the table. For the plots, the sequences of metrics values are truncated at the first occurrence of inf/nan.


.. _reproducibility:

Reproducibility
---------------
Many decentralized algorithms and network scenarios have stochastic features. An example are algorithms that make use of
stochastic gradients computed on a subset :math:`\mathcal{B} \subset \{ 1, \ldots, m_i \}` of the available data:

.. math::
    \hat{\nabla} f_i(x) = \frac{1}{|\mathcal{B}|} \sum_{h \in \mathcal{B}} \ell(x, d_i^h).

See instead :doc:`this page <customizing>` for how to create networks with stochastic features.

The following example creates the same linear regression problem as before, but setting ``batch_size=2``, which means
that agents compute stochastic gradients with two datapoints.

.. literalinclude:: ../../examples/stochastic_fed_example.py
    :language: python
    :linenos:

Since the datapoints to use during computation are chosen at random, the algorithms are now stochastic. In order to run
reproducible simulations, one can use :func:`~decent_bench.utils.interoperability.set_seed` to set a seed before the
benchmark is run.

Additionally, the benchmark runs several trials (``n_trials=10`` in :func:`~decent_bench.benchmark.benchmark`) and
averages across them, to provide more informative results. Importantly, a different seed
*derived deterministically from the main seed* is used for each trial. This means that each trial is distinct (making
aggregation across trials meaningful), while maintaining reproducibility. The following figures show the results with
and without stochastic gradients.

.. list-table::
   :widths: 1 1

   * - .. figure:: ../_static/basic_fed_example_plots.png
          :align: center

          Using full gradients
     - .. figure:: ../_static/stochastic_fed_example.png
          :align: center

          Using stochastic gradients

The plots on the left are fully deterministic, which means that even if multiple trials are run, their outputs will
be exactly the same. The plots on the right, instead, are stochastic, and different trials yield different results.
This means that when aggregating over trials, there is some variation. This is highlighted by plotting the mean as a
solid line, and the minimum and maximum as the envelope around it. This is done by default by decent-bench.


Note:
    How is it possible to set a seed? The functionality to set a seed is provided by the interoperability package
    :mod:`decent_bench.utils.interoperability`. This package provides a wrapper around a number of widely used
    frameworks (NumPy, PyTorch, TensorFlow, JAX), exposing a common API for all of them. Implementing all parts of the
    benchmarking pipeline using the functions in the interoperability API allows to implement each item once, while
    allowing to select different framework for the backend. ``set_seed`` is part of this interoperability API,
    and it takes care of setting the seed for whichever framework is selected. More details on interoperability in :doc:`this page <customizing>`.
    The seed is set for the built-in python ``random`` module as well.


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
   
.. literalinclude:: ../../examples/checkpointing_fed_custom_options.py
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

.. code-block:::: python

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
