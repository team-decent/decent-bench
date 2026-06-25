Utils
=====


Drawing networks
----------------
Networks, both :class:`~decent_bench.networks.FedNetwork` and :class:`~decent_bench.networks.P2PNetwork`, can be drawn
using `NetworkX <https://networkx.org/en/>`_ drawing tools. To do so, decent-bench exposes
:func:`~decent_bench.utils.network_utils.plot_network`, as shown in the following example.


.. literalinclude:: ../../../docs/examples/network_drawing.py
    :language: python
    :linenos:


Generating a network from its adjacency matrix
----------------------------------------------
`NetworkX <https://networkx.org/en/>`_ graphs, which define the topology of :class:`~decent_bench.networks.P2PNetwork`
instances, can also be created by their adjacency matrices, as shown in the following example.


.. literalinclude:: ../../../docs/examples/network_from_adjacency.py
    :language: python
    :linenos:


Computing the linear convergence rate
-------------------------------------
When algorithms are know to converge linearly, it is possible to estimate their convergence rate using the utility
:func:`~decent_bench.metrics.utils.linear_convergence_rate`, as shown in the following example.

.. literalinclude:: ../../../docs/examples/linear_convergence_rate.py
    :language: python
    :linenos:
