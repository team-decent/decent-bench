.. _reproducibility:

Reproducibility
---------------
Many decentralized algorithms and network scenarios have stochastic features. An example are algorithms that make use of
stochastic gradients computed on a subset :math:`\mathcal{B} \subset \{ 1, \ldots, m_i \}` of the available data:

.. math::
    \nabla f_i(x) \approx \frac{1}{|\mathcal{B}|} \sum_{h \in \mathcal{B}} \ell(x, d_i^h).

See instead :doc:`this page </user/customizing/network>` for how to create networks with stochastic features.

The following example creates the same linear regression problem as before, but setting ``batch_size=2``, which means
that agents compute stochastic gradients with two datapoints.


.. literalinclude:: ../../../examples/stochastic_fed_example.py
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

   * - .. figure:: ../../_static/basic_fed_example_plots.png
          :align: center

          Using full gradients
     - .. figure:: ../../_static/stochastic_fed_example.png
          :align: center

          Using stochastic gradients


The plots on the left are fully deterministic, which means that even if multiple trials are run, their outputs will
be exactly the same. The plots on the right, instead, are stochastic, and different trials yield different results.
This means that when aggregating over trials, there is some variation. This is highlighted by plotting the mean as a
solid line, and the minimum and maximum as the envelope around it. This is done by default by decent-bench.


.. _interop_seed:

.. note::
    How is it possible to set a seed? The functionality to set a seed is provided by the interoperability package
    :mod:`decent_bench.utils.interoperability`. This package provides a wrapper around a number of widely used
    frameworks (NumPy, PyTorch, TensorFlow, JAX), exposing a common API for all of them. Implementing all parts of the
    benchmarking pipeline using the functions in the interoperability API allows to implement each item once, while
    allowing to select different framework for the backend. ``set_seed`` is part of this interoperability API, and it
    takes care of setting the seed for whichever framework is selected. More details on interoperability in
    :doc:`this page </user/customizing/algorithm>`. The seed is set for the built-in python ``random`` module as well.
