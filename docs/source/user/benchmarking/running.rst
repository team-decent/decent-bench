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

.. literalinclude:: ../../../examples/basic_fed_example.py
    :language: python
    :linenos:

With table results:

.. code-block:: text

    algorithm                   FedAvg             Scaffold                                                                                                                                                                                                                                    
    metric                                                                                                                                                                                                                                                                                     
    gradient norm  3.71e-01 ± 0.00e+00  2.20e-14 ± 0.00e+00                                                                                                                                                                                                                                    
    x error        3.16e-01 ± 0.00e+00  1.42e-14 ± 0.00e+00  

and plots:

.. image:: ../../_static/basic_fed_example_plots.png
    :align: center


Peer-to-peer example
^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../examples/basic_p2p_example.py
    :language: python
    :linenos:

With table results:

.. code-block:: text

    algorithm                     ADMM                  DGD                   ED                                                                                                                                                                                                               
    metric                                                                                                                                                                                                                                                                                     
    gradient norm  3.62e-14 ± 0.00e+00  1.60e-01 ± 0.00e+00  1.65e-08 ± 0.00e+00                                                                                                                                                                                                               
    x error        2.84e-14 ± 0.00e+00  1.90e-01 ± 0.00e+00  1.97e-08 ± 0.00e+00 

and plots:

.. image:: ../../_static/basic_p2p_example_plots.png
    :align: center


Explanation
^^^^^^^^^^^
In both examples, we generate a linear regression problem

.. literalinclude:: ../../../examples/basic_p2p_example.py
    :language: python
    :lines: 15

characterized by the local costs
:math:`f_i(x_i) = \frac{1}{m_i} \sum_{h = 1}^{m_i} \frac{1}{2} (a_i^h x_i - b_i^h)^2` with feature vectors
:math:`a_i^h \in \mathbb{R}^{1 \times n}` and targets :math:`b_i^h \in \mathbb{R}`. Since the costs are quadratic, the
``create_regression_problem`` utility also computes the optimal solution ``x_optimal``.

We then create the federated or peer-to-peer network of agents to solve the problem, respectively:

.. literalinclude:: ../../../examples/basic_fed_example.py
    :language: python
    :lines: 14

.. literalinclude:: ../../../examples/basic_p2p_example.py
    :language: python
    :lines: 17-18

where each agent is assigned one of the cost functions :math:`f_i` contained in ``costs``.

These steps set up the benchmark problem, which is represented by the data structure:

.. literalinclude:: ../../../examples/basic_fed_example.py
    :language: python
    :lines: 15

The next steps are the execution of the benchmark using the :func:`~decent_bench.benchmark.benchmark` function, passing
a list of the algorithms to be tested as ``algorithms``, each with its hyperparameters. The results are contained in the
:class:`~decent_bench.benchmark.BenchmarkResult` object (``results`` in the examples), which can be used to compute the
performance metrics.

In the examples, two performance metrics are selected (which are instances of :class:`~decent_bench.metrics.Metric`):

.. literalinclude:: ../../../examples/basic_fed_example.py
    :language: python
    :lines: 32

with :class:`~decent_bench.metrics.metric_library.XError` being the distance from the optimal solution, and
:class:`~decent_bench.metrics.metric_library.GradientNorm` the norm of the gradient of :math:`\sum_{i = 1}^N f_i`. If
the ``table_metrics`` and ``plot_metrics`` arguments of :func:`~decent_bench.benchmark.compute_metrics` are not specified,
all the available metrics are used instead. Not all metrics are always available: for example
:class:`~decent_bench.metrics.metric_library.XError` is not available if ``x_optimal`` is not available.

.. note::
    In these example scripts, execution is guarded with ``if __name__ == "__main__":`` to avoid potential issues with
    multi-threading and some of the dependencies. Errors and unexpected results might appear if this guard is not used.

The following section explains in more detail how to customize the benchmark workflow.