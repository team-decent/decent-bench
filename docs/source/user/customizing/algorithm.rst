Algorithms
----------

This section discusses available algorithms and how to implement algorithms from scratch, in particular how to ensure
that they satisfy interoperability requirements.


Algorithm structure
^^^^^^^^^^^^^^^^^^^
Algorithms are subclasses of :class:`~decent_bench.algorithms.federated.FedAlgorithm` and
:class:`~decent_bench.algorithms.p2p.P2PAlgorithm`, depending on the network architecture selected for the
benchmark problem. In turn these are subclasses of :class:`~decent_bench.algorithms.Algorithm` which is shown below:

.. literalinclude:: ../../../../decent_bench/algorithms/_algorithm.py
    :language: python
    :lines: 8-13, 33-60, 83-125

Algorithms are thus characterized by the following:

* ``iterations``: the number of iterations of the algorithm that should be run during benchmarking.
* ``name``: a representative name (usually the algorithm acronym); algorithms tested during a benchmark run must have unique names.
* ``__post_init__``: hyperparameter validation can optionally be performed here; if an algorithm implements this method, it will be automatically called after init.
* ``run``: this method calls ``initialize`` first, then ``step`` for the specified number of iterations. ``run`` must not be modified, only ``initialize`` and ``step``.


Available algorithms
^^^^^^^^^^^^^^^^^^^^
Currently available algorithms are:

* *Federated*: :tagged:`algorithm, federated`.
* *Peer-to-peer*: :tagged:`algorithm, peer-to-peer`.


.. note:: The implementation of these algorithms might be slighlty modified w.r.t. the original papers, in particular
    for algorithms that were not explicitly designed to handle asynchronous activation or communication drops. Indeed,
    *the objective is to provide a library of algorithms that can be deployed in the realistic scenarios characterized in
    :doc:`/user/customizing/network`*. To do so, some changes were made, like using only
    :func:`~decent_bench.networks.Network.active_agents` during ``step``.


Details on federated algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Federated algorithms follow the same general pattern discussed above, but in addition expose the
:func:`~decent_bench.algorithms.federated.FedAlgorithm.aggregate` method and require passing a
:class:`~decent_bench.schemes.ClientSelectionScheme` as the argument ``selection_scheme``.
Indeed, federated algorithms usually apply *client selection* (a.k.a. *partial participation*), by which the server
selects only a subsection of the currently active clients for a round of communication (clients send the local models
to the server, which averages them).

Different :class:`~decent_bench.schemes.ClientSelectionScheme` implementations are available to model different
strategies, see :mod:`~decent_bench.schemes`.

.. note:: Aggregation is another place where algorithms might differ from the original papers, or rather, the
    original implementations. Indeed, most papers present federated algorithms as using averaging during the aggregation
    step; while the code associated to the papers often uses weighted averaging based on *e.g.* the number of
    datapoints available at each client. In decent-bench, we have decided to implement the algorithms as they are
    presented in the papers' pseudocode.


The following shows an example benchmarking federated algorithms with a fair client selection scheme (which prioritizes
clients that were selected fewer times).


.. literalinclude:: ../../../examples/client_selection.py
    :language: python
    :linenos:


Implementing algorithms from scratch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Implementing a new algorithm from scratch requires the following:

* **Init**: the algorithm should have as init arguments (in this order): ``iterations``, any hyperparameters (like step-size), initial conditions (like ``x0`` for :math:`x_0`). decent-bench offers some utilities for common initialization choices: :func:`~decent_bench.algorithms.utils.normal_initialization`, :func:`~decent_bench.algorithms.utils.uniform_initialization`, :func:`~decent_bench.algorithms.utils.pytorch_initialization`. Zero initialization is the default when *e.g.* ``x0 = None``.
* ``__post_init__``: for custom hyperparameter validation.
* **Initialize**: during which the states of each agent are initialized, including :attr:`~decent_bench.agents.Agent.x` and any auxiliary variables. :attr:`~decent_bench.agents.Agent.x` represents the local state, for example the parameters of the locally trained model.
* **Step**: which defines all the steps of the algorithm to be performed during one iteration. This is usually the content of the main for-loop in papers' pseudocode. The step *must* update active agents's :attr:`~decent_bench.agents.Agent.x` states *exactly once*, since the benchmark pipeline assumes that this is the case. As discussed above, ``step`` should also rely (uniquely) on the methods exposed by network object, like :meth:`~decent_bench.networks.Network.active_agents`. This is to ensure that the benchmark setting is correctly represented.

The following is an example of a peer-to-peer algorithm already implemented in decent-bench.

.. literalinclude:: ../../../../decent_bench/algorithms/p2p/_ed.py
    :language: python
    :linenos:
    :lines: 13-75


It is worth pointing out some important implementation choices:

* During ``step``, each operation that is performed in parallel by the active agents is contained in a separate for-loop. In the case of ED, there is one for-loop for updating auxialiary variables and one for updating :attr:`~decent_bench.agents.Agent.x`. Dividing in for-loops is mandatory to ensure consistency with papers' pseudocode.
* The algorithm uses :func:`~decent_bench.networks.Network.active_agents` to iterate only over the available agents, and :func:`~decent_bench.networks.P2PNetwork.broadcast` (which employs :func:`~decent_bench.networks.Network.send`), to make sure that the network scenario is correctly simulated.
* Each agent stores a dictionary of auxiliary variables :attr:`~decent_bench.agents.Agent.aux_vars` that can be used to store other variables that are not :attr:`~decent_bench.agents.Agent.x`. Additionally, the messages received by an agent can be accessed via :func:`~decent_bench.agents.Agent.messages`, which is populated by the network every time a message is sent (provided it is not dropped).



.. _interoperability:

Interoperability
^^^^^^^^^^^^^^^^
The goal of decent-bench is to provide a benchmarking pipeline (from networks to algorithms) that is not
tied to any single computational framework. This is accomplished via the interoperability layer
:mod:`~decent_bench.utils.interoperability`, exposing a standard API which then interfaces with the supported
frameworks (currently, NumPy, PyTorch, TensorFlow, JAX).

The idea then is to use the interoperability layer while defining network and algorithms, ensuring that they can be
reused across frameworks, instead of defining framework-specific versions of each algorithm.

The components of the interoperability layer include, first the :class:`~decent_bench.utils.array.Array` object, which wraps
a framework-native array/tensor. The benchmarking pipeline assumes that only :class:`~decent_bench.utils.array.Array`
objects are passed around. The second component is the interoperability API, which defines functions like
:func:`decent_bench.utils.interoperability.zeros` and :func:`decent_bench.utils.interoperability.eye_like` (used in
the ED algorithm shown above).

Algorithms, networks, and schemes should therefore always use interoperability functions. When they are not available, 
a workaround is to convert to NumPy (:class:`decent_bench.utils.interoperability.to_numpy`) and then converting
back to :class:`~decent_bench.utils.array.Array`.

Costs are the only place where framework-native operations should be employed, making sure to use
:func:`~decent_bench.utils.interoperability.autodecorate_cost_method` to correctly interface with the
interoperability layer (see discussion :ref:`here <interop_cost>`).


Advanced algorithm implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The following is a short example showing how to implement more efficient custom algorithms. The example shows a version
of :class:`~decent_bench.algorithms.p2p.LT_ADMM` customized for :class:`decent_bench.costs.PyTorchCost` by using
a PyTorch optimizer during local training. Notice that the initialize method sets up the PyTorch optimizer for each
agent, and the ``_local_training`` method performs local training using the optimizer.

.. code-block:: python

    from dataclasses import dataclass
    from typing import TYPE_CHECKING, Any

    import decent_bench.utils.interoperability as iop
    from decent_bench.agents import Agent
    from decent_bench.algorithms.p2p import LT_ADMM
    from decent_bench.costs import PyTorchCost
    from decent_bench.networks import P2PNetwork

    if TYPE_CHECKING:
        import torch

    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "PyTorch is required for LT-ADMM algorithm, but it is not installed. "
            "Please install PyTorch to use this algorithm."
        ) from e


    @dataclass(eq=False)
    class LT_ADMM_TORCH(LT_ADMM):
        opt_cls: type[torch.optim.Optimizer] | None = None  # PyTorch optimizer class to use for local training, e.g. torch.optim.Adam
        opt_kwargs: dict[str, Any] | None = None  # Keyword arguments for PyTorch optimizer
        sched_cls: type[torch.optim.lr_scheduler.LRScheduler] | None = None  # PyTorch scheduler class for local training
        sched_kwargs: dict[str, Any] | None = None  # Keyword arguments for PyTorch scheduler
        name: str = "LT-ADMM-TORCH"

        def initialize(self, network: P2PNetwork) -> None:
            super().initialize(network)
            for i in network.agents():
                if not isinstance(i.cost, PyTorchCost):
                    raise TypeError(f"LT-ADMM-TORCH requires PyTorchCost, but agent {i} has cost of type {type(i.cost)}")

                # Initialize PyTorch optimizer for local training if use_torch_optim is True
                if self.opt_cls is not None:
                    if self.opt_kwargs is None:
                        self.opt_kwargs = {}
                    self.opt_kwargs.setdefault("lr", self.step_size)
                    i.cost.init_local_training(
                        opt_cls=self.opt_cls,
                        opt_kwargs=self.opt_kwargs,
                        sched_cls=self.sched_cls,
                        sched_kwargs=self.sched_kwargs,
                    )

        def _local_training(self, agent: Agent, network: P2PNetwork) -> None:
            if TYPE_CHECKING:
                if not isinstance(agent.cost, PyTorchCost):
                    raise TypeError(
                        f"LT-ADMM-TORCH requires PyTorchCost, but agent {agent} has cost of type {type(agent.cost)}"
                    )

            agent.aux_vars["phi"] = iop.copy(agent.x)
            z_sum = iop.sum(agent.aux_vars["z_i"], dim=0)
            multiplier = self.penalty * len(network.neighbors(agent))
            correction = self.aux_step_size * (multiplier * agent.x - z_sum)

            if self.opt_cls is not None:  # Use PyTorch optimizer
                agent.aux_vars["phi"] = agent.cost.local_training(
                    x=agent.aux_vars["phi"],
                    iterations=self.num_local_steps,
                    regularization=correction,
                    agent=agent,
                )
            else:  # Default to gradient descent
                for _ in range(self.num_local_steps):
                    current_gradient = agent.cost.gradient(agent.aux_vars["phi"])
                    step = self.step_size * current_gradient + correction
                    # Update phi_i,k according to gradient step
                    agent.aux_vars["phi"] -= step

            # Update agent's main parameter
            agent.x = agent.aux_vars["phi"]
