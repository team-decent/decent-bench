.. _introduction:

Introduction
============

The success of modern artificial intelligence (AI) is fueled by access to massive datasets and efficient training
methods. However, in many use cases it is not possible to apply centralized training, for example because of
privacy (*e.g.* diagnostic models for healthcare) or practical constraints (*e.g.* search and rescue with a
multi-robot system).
Therefore a suitable alternative is *decentralized learning*, in which agents connected in a network cooperatively train
a model without the need to share data.

Two main decentralized learning architectures have been explored: *federated* :cite:p:`Survey_FED_Gafni, Survey_FED_Li`
and *peer-to-peer* (a.k.a. *distributed*, *decentralized federated*) :cite:p:`Survey_P2P_Notarstefano, Survey_P2P_Yang`,
as depicted below.

.. list-table::
   :widths: 1 1

   * - .. figure:: ../_static/architecture-federated.png
          :align: center

          Federated architecture
     - .. figure:: ../_static/architecture-distributed.png
          :align: center

          Peer-to-peer architecture

In particular:

* A **federated architecture** is characterized by a central agent, called *coordinator* or *server*, connected to a set of *clients* or *agents*. The clients store data and process it to train local models, send them to the coordinator, which aggregates all models into a global one and broadcasts back to the clients.
* A **peer-to-peer architecture** instead is characterized by agents sharing locally trained models directly with neighboring agents, and aggregate the received models.


Problem formulation
-------------------
Formally, decentralized training of a model requires the solution of a constrained optimization problem like:

.. math::
   :label: decentralized-problem

   &\text{min} \sum_{i = 1}^N f_i(x_i) \\ \text{s.t.} & \ x_1 = x_2 = \ldots = x_N

where :math:`N` is the number of agents, :math:`x_i \in \mathbb{R}^n` contains all the parameters of the model being
trained, and the local *cost* or *loss* functions :math:`f_i : \mathbb{R}^n \to \mathbb{R} \cup \{ +\infty \}` have
and empirical risk minimization structure:

.. math::

   f_i(x_i) = \sum_{h = 1}^{m_i} \ell(x_i, d_i^h)

where :math:`d_i^h` are the :math:`m_i` datapoints of agent :math:`i` (*e.g.* pairs of feature vector and label)
and `ell` is a loss function (*e.g.* squared error, 0-1 loss).

Beyond learning, a wide range of decentralized tasks can be formulated as :eq:`decentralized-problem` with suitable
cost functions and, potentially, the addition of other constraints. The following section briefly discusses some
use cases in which we need to solve :eq:`decentralized-problem`.


Use cases
---------
* **Learning**: consider a set of hospitals which aim to train diagnostic models based on the patients records they have access to. Directly sharing the data with each other would constitute a privacy breach, and thus they need to resort to decentralized learning.
* **Exploration**: consider a set of mobile robots tasked with exploring and mapping an environment (*e.g.* for the purpose of search and rescue). The robots can establish direct communication links with each other to coordinate exploration in a decentralized fashion.
* **Resource allocation**: consider a set of power generators linked to the power grid, each with different output (*e.g.* different types of renewable/fossil energy). These generators need to coordinate in order to fully satisfy the power demand in the grid.


Practical challenges
--------------------
Achieving decentralized tasks thus boils down to designing tailored optimization algorithms to solve :eq:`decentralized-problem`.
However, this is far from a simple objective. Indeed, in most use cases algorithms need to operate under a set of
practical constraints, which include:

* **Limited computational power**, *e.g* if they are mobile robots.
* **Limited communication**, *e.g.* if they rely on wireless links with limited bandwidth.
* **Heterogeneous resources**, *e.g.* different agents have different computational power.
* **Heterogenous data**, *e.g.* each agent has different sensors to collect data.

It is therefore necessary to account for these challenges at the design stage.


Why decent-bench
------------------
The objective of decent-bench (for *decentralized-benchmarking*) is to support the design of decentralized algorithms that can be deployed in a wide range
of challenging scenarios. In particular, decent-bench offers the following functionalities:

* **Simulating deployment scenarios**: decent-bench allows to define realistic scenarios in which to test decentralized algorithms (both federated and peer-to-peer); this includes simulating unreliable communications and limited/heterogeneous computational power.
* **Benchmarking**: decent-bench defines a simple benchmarking pipeline to test and compare different decentralized algorithms in the same scenario. Additionally, it provides a library of state-of-the-art algorithms ready to use.
* **Reproducibility**: as part of the benchmarking pipeline, decent-bench allows setting random seeds to ensure that results are easily reproducible.
* **Paper-ready results**: decent-bench provides the results of a benchmark run in a format that can be directly pasted into papers (with several customization options). The raw results are also available as ``pandas.DataFrame`` for easy inspection.


References
----------

.. bibliography::
   :cited: