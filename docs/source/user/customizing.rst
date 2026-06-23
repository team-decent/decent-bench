Customizing
===========

This page describes where to customize each part of a benchmark pipeline.

Network
-------

You can customize topology and communication constraints when building a benchmark problem.

Common options include:

- network structure (for example random regular graphs)
- asynchronous activation
- compression
- noise
- packet drops

.. literalinclude:: ../../../test/user-guide/customizing_network.py
   :language: python

Algorithm
---------

Use built-in algorithms from :mod:`~decent_bench.distributed_algorithms`, or implement your own by subclassing
:class:`~decent_bench.distributed_algorithms.Algorithm`.

When implementing a custom algorithm, provide:

- ``initialize(network)``
- ``step(network, iteration)``
- optional ``finalize(network)``

Always update agent states through the network and interoperability abstractions so metrics and communication schemes
remain consistent.

Cost Function
-------------

You can use built-in costs or implement a custom cost by subclassing :class:`~decent_bench.costs.Cost`.

Cost composition supports arithmetic patterns such as:

- ``cost + regularizer``
- ``scalar * cost``
- ``sum(costs)``

For empirical objectives, compose carefully so batch-aware behavior is preserved.

Benchmark Pipeline
------------------

A practical customization progression is:

1. Start with an out-of-the-box problem constructor.
2. Modify selected fields on the generated problem.
3. Build a full :class:`~decent_bench.benchmark.BenchmarkProblem` from custom resources when needed.

This workflow keeps iteration speed high early, while still allowing full control for advanced experiments.

Metrics
-------

Customize metrics at two levels:

- post-run table/plot metrics passed to :func:`~decent_bench.benchmark.compute_metrics`
- runtime metrics passed to :func:`~decent_bench.benchmark.benchmark`

You can create your own metrics by subclassing :class:`~decent_bench.metrics.Metric` or
:class:`~decent_bench.metrics.RuntimeMetric`.