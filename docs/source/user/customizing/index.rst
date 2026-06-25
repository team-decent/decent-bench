Customizing
===========

Recalling the simple example below, which was discussed on :doc:`this page </user/benchmarking/running>`, we can highlight several
components of a benchmark workflow:

* **Network**, either federated or peer-to-peer.
* **Costs** assigned to each one of the agents (a.k.a. clients).
* **Algorithms**, characterized by a choice of hyperparameters.
* **Metrics** used to evaluate and compare performance.

.. literalinclude:: ../../../examples/basic_fed_example.py
    :language: python
    :linenos:

The following sections will discuss each of these components, showing how pick them from the library of implemented
items provided by decent-bench or how to implement them from scratch.


Index
-----

.. toctree::
   :maxdepth: 1

   network
   cost
   algorithm
   metric