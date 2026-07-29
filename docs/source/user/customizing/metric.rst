Metrics
-------

There are two types of performance metrics: :class:`~decent_bench.metrics.Metric`, which are used in 
:func:`~decent_bench.benchmark.compute_metrics`, and :class:`~decent_bench.metrics.RuntimeMetric`, which can
be displayed during execution of :func:`~decent_bench.benchmark.benchmark` (especially for very time-consuming
simulations).

decent-bench already offers a wide number of :class:`~decent_bench.metrics.Metric` objects in
:mod:`~decent_bench.metrics.metric_library`, and the runtime metrics :tagged:`runtime metric`. If more metrics are
needed, they can be straighforwardly implemented as subclasses of :class:`~decent_bench.metrics.Metric` and
:class:`~decent_bench.metrics.RuntimeMetric`, providing a concrete implementation of the ``compute`` method.
For :class:`~decent_bench.metrics.Metric`, the ``is_available`` method can also be implemented, which signals when
a metric can be computed (*e.g.* :class:`~decent_bench.metrics.metric_library.XError` is only available
when ``x_optimal`` for the benchmark problem was computed).


.. note:: Runtime metrics are not meant to be precise, only to provide some insight into how the simulations
    are progressing (*e.g.* is any algorithm diverging due to a poor hyperparameters choice?). Therefore, when
    implementing new :class:`~decent_bench.metrics.RuntimeMetric` objects, prioritize computational efficiency to
    precision. Precise performance evaluation is executed through computation of :class:`~decent_bench.metrics.Metric`
    instances.
