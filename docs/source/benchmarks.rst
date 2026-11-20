Benchmarks
==========

decent-bench includes some benchmarks that are already implemented in the ``examples`` folder available
`here <https://github.com/team-decent/decent-bench/tree/main/examples>`_. The benchmarks are listed below
with a short description of the use case and features.


Using benchmarks
----------------
Benchmarks are provided as sub-folders of the `examples <https://github.com/team-decent/decent-bench/tree/main/examples>`_
folder. Each benchmark consists of a Jupyter notebook and additional code in the `src` folder, and is self-contained.
Benchmarks can be customized by:

#. Adding custom algorithms to test, which are simply passed as arguments of :func:`~decent_bench.benchmark.benchmark`.
#. Customizing the benchmark settings through the arguments defined at the top of the notebooks.


Available benchmarks
--------------------
The following list includes the currently available benchmarks, specifying to which architectures they apply.

* ``consensus`` [peer-to-peer]: this benchmark tests a simple *average consensus* problem, where the agents need to collaboratively compute the average of their initial observations :math:`u_i`.
* ``FEMNIST`` [federated]: this benchmark tests federated algorithms for a classification task of handwritten characters from the FEMNIST dataset.
* ``MNIST`` [peer-to-peer]: this benchmark tests peer-to-peer algorithms for a classification task of handwritten digits from the MNIST dataset.
* ``nim`` [peer-to-peer]: this benchmark tests peer-to-peer algorithms for *neural implicit mapping*, where the agents are spread over an environment they need to map, and collaboratively train a neural network approximating the density of the environment (which pinpoints where obstacles are).
