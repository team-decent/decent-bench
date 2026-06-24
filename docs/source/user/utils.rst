Utils
=====

This page highlights utility components that are commonly useful when running and debugging benchmarks.


Network Utilities
-----------------

Use :mod:`~decent_bench.utils.network_utils` to visualize network topologies during debugging or reporting.

.. literalinclude:: ../../../test/user-guide/utils_network_plot.py
    :language: python

Algorithm Helpers
--------------------------------------

Two helper areas are especially useful for custom extensions:

- :mod:`~decent_bench.utils.algorithm_helpers` for common initialization patterns





Network utilities
-----------------
Plot a network explicitly when you need it:

.. code-block:: python

    import networkx as nx
    from decent_bench import benchmark
    from decent_bench.utils import network_utils
    from decent_bench.costs import LinearRegressionCost

    problem = benchmark.create_regression_problem(LinearRegressionCost, n_agents=25, n_neighbors_per_agent=3)

    # Plot using decent-bench helper (wraps :func:`networkx.drawing.nx_pylab.draw_networkx`)
    network_utils.plot_network(problem.network_structure, layout="circular", with_labels=True)

    # Or call NetworkX directly on the graph
    pos = nx.drawing.layout.spring_layout(problem.network_structure)
    nx.drawing.nx_pylab.draw_networkx(problem.network_structure, pos=pos, with_labels=True)

For more options, see the `NetworkX drawing guide <https://networkx.org/documentation/stable/reference/drawing.html>`_.
