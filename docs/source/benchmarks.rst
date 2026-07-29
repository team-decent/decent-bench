Benchmarks
==========

decent-bench includes some benchmarks that are already implemented in the ``examples`` folder available
`here <https://github.com/team-decent/decent-bench/tree/main/examples>`_. The benchmarks are listed below
with a short description of the use case and features.


How to use benchmarks
---------------------
Benchmarks are implemented as Jupyter notebooks, and include all the code to set up and run the benchmark in a
self-contained way. Testing a new algorithm on the benchmark is as easy as adding a new item to the list ``algorithms``
that is passed as argument of :func:`~decent_bench.benchmark.benchmark`. Each notebook also includes a number
of arguments definitions that can be used to customize the benchmark problem.


Available benchmarks
--------------------
The follwing list includes the currently available benchmarks, specifying to which architectures they apply.

* ``consensus`` [peer-to-peer]: this benchmark tests a simple *average consensus* problem, where the agents need to collaboratively compute the average of their initial observations :math:`u_i`.
* ``FEMNIST`` [federated]: this benchmark tests federated algorithms for a classification task of handwritten characters from the FEMNIST dataset.
* ``MNIST`` [peer-to-peer]: this benchmark tests peer-to-peer algorithms for a classification task of handwritten digits from the MNIST dataset.
* ``nim`` [peer-to-peer]: this benchmark tests peer-to-peer algorithms for *neural implicit mapping*, where the agents are spread over an environment they need to map, and collaboratively train a neural network approximating the density of the environment (which pinpoints where obstacles are).
