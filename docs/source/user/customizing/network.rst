Networks
--------

The network is the first fundamental choice, responsible for defining the realistic scenario in which
the benchmark should be run.


Architecture
^^^^^^^^^^^^
.. list-table::
   :widths: 1 1

   * - .. figure:: ../../_static/architecture-federated.png
          :align: center

          Federated architecture
     - .. figure:: ../../_static/architecture-distributed.png
          :align: center

          Peer-to-peer architecture


decent-bench currently supports federated and peer-to-peer networks, which are characterized as follows:

* :class:`~decent_bench.networks.FedNetwork`: is defined by passing a list of :class:`~decent_bench.agents.Agent` objects characterizing the clients, and optionally one for the server. If the server is not specified, a default instance is generated, which fits most use cases.
* :class:`~decent_bench.networks.P2PNetwork`: is defined by passing a `NetworkX <https://networkx.org/en/>`_ graph defining the topology, and a list of :class:`~decent_bench.agents.Agent` objects defining the agents.

Both networks internally store the graph topology as a `NetworkX <https://networkx.org/en/>`_ graph with the
:class:`~decent_bench.agents.Agent` instances as nodes. :class:`~decent_bench.networks.FedNetwork` and
:class:`~decent_bench.networks.P2PNetwork` then expose a number of methods to inspect the graph and, importantly,
method :func:`~decent_bench.networks.Network.send` *to simulate realistic communications across the network*,
discussed in the next section.

Customizing the network thus can be done by 1) selecting the architecture, 2) defining a number of agents and, for the
peer-to-peer case, a topology. Owing to the use of `NetworkX <https://networkx.org/en/>`_ as the representation of the
network architecture, the peer-to-peer topology can be defined using `NetworkX <https://networkx.org/en/>`_ functions,
as in the following example which generates three networks with complete, star, and ring topologies.

.. literalinclude:: ../../../examples/network_generation.py
    :language: python
    :linenos:

.. note:: By design, one :class:`~decent_bench.agents.Agent` object can only be assigned to one network. This is because
    :class:`~decent_bench.agents.Agent` instances store information as the benchmark progresses, and assigning the same
    instance to two networks might lead to conflicts. In most use cases, a single network is involved and this is not a concern.


Realistic communications
^^^^^^^^^^^^^^^^^^^^^^^^
As briefly discussed in :doc:`the introduction </user/introduction>`, when deploying decentralized algorithms we are faced
with several practical challenges. These include:

* **Noisy communications**: *e.g.* due to wireless interference.
* **Compressed communications**: if the bandwith is limited, compressing communications is necessary; this, however, results in some loss of information.
* **Communications drops**: due to network issues, communications might fail to reach the recipient.

These communications constraints are defined as "Scheme" objects in decent-bench, collected in 
:mod:`~decent_bench.schemes`. Instances of schemes are passed into the network constructors as the arguments:

* ``message_noise``: defined as instances of :class:`~decent_bench.schemes.NoiseScheme`.
* ``message_compression``: defined as instances of :class:`~decent_bench.schemes.CompressionScheme`.
* ``message_drop``: defined as instances of :class:`~decent_bench.schemes.DropScheme`.

These arguments allow passing either one instance only, which applies the same scheme to all agents, or a dictionary
keyed by agent to specify per-agent schemes. This allows simulating realistic scenarios where the agents have
*heterogeneous resources*.

When :func:`~decent_bench.networks.Network.send` is called, all of these communication schemes are applied to the
message being communicated: the message is compressed, then noise is added to it, and finally the message is delivered
only if it is not dropped.

decent-bench takes care of simulating the communication constraints, so defining the schemes is the only thing needed.
One can also create custom schemes: this is done by creating an implementation of the abstract schemes
:class:`~decent_bench.schemes.NoiseScheme`, :class:`~decent_bench.schemes.CompressionScheme`,
:class:`~decent_bench.schemes.DropScheme`. Below is an example of ``DropScheme`` implementation, and see
:mod:`~decent_bench.schemes` for the full library.


.. literalinclude:: ../../../../decent_bench/schemes.py
    :language: python
    :lines: 688-708


Asynchronous agent activation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In practical scenarios, agents might be equipped with heterogeneous computational resources. This means that they
complete local data processing at different frequencies. When an agent complese local processing it activates, by sending
the newly trained local model to the server (in federated architectures) or to neighbors (in peer-to-peer).

To simulate *asynchronous activation* of agents, decent-bench provides :class:`decent_bench.schemes.AgentActivationScheme`.
At each iteration, the network polls all agents to check if they are active (*i.e.* ready to communicate) or inactive
(*i.e.* still processing). Then, only active agents will participate in communications. The following shows a concrete
implementation of :class:`decent_bench.schemes.AgentActivationScheme`.


.. literalinclude:: ../../../../decent_bench/schemes.py
    :language: python
    :lines: 45-65


Instances of :class:`decent_bench.schemes.AgentActivationScheme` are passed directly to the init of
:class:`~decent_bench.agents.Agent` via the ``activation`` argument. Then, networks expose the method
:func:`~decent_bench.networks.Network.active_agents` which returns a list of active agents
for the current iteration, as determined by their activation schemes. This method is then used in algorithms to
select agents for the next round of training, see :doc:`/user/customizing/algorithm`.


.. note:: The server in :class:`~decent_bench.networks.FedNetwork` is always active
    (with scheme :class:`decent_bench.schemes.AlwaysActive`).


.. note:: Another source of partial agent participation in federated settings is *client selection*,
    discussed in :doc:`/user/customizing/algorithm`. But where agent activation is internally determined by
    :class:`~decent_bench.agents.Agent` instances, client selection is determined externally by the server.


Example
^^^^^^^
The following is a full example showing the creation of a :class:`~decent_bench.networks.P2PNetwork` with all
possible realistic constraints.

.. literalinclude:: ../../../examples/network_generation_schemes.py
    :language: python
    :linenos: