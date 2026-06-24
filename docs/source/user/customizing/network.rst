Network
-------

You can customize topology and communication constraints when building a benchmark problem.

Common options include:

- network structure (for example random regular graphs)
- asynchronous activation
- compression
- noise
- packet drops

.. literalinclude:: ../../../test/user-guide/customizing_network.py
   :language: python




Network constraints
^^^^^^^^^^^^^^^^^^^
Networks can model constrained communication through activation, compression,
noise, and message-drop schemes. These schemes are configured when creating a
network or benchmark problem and are applied automatically during
:meth:`~decent_bench.networks.Network.send`.

Activation schemes control whether an agent is available at a given iteration.
The built-in options are :class:`~decent_bench.schemes.AlwaysActive`,
:class:`~decent_bench.schemes.UniformActivationRate`,
:class:`~decent_bench.schemes.MarkovChainActivation`, and
:class:`~decent_bench.schemes.PoissonActivation`.
:class:`~decent_bench.schemes.CyclicActivation` alternates between active and
inactive intervals, with an optional phase offset to model staggered recurring
availability windows.
Implemented federated algorithms in decent-bench first ask the network for
active clients and then apply the client-selection scheme to that active set.

Compression schemes transform messages before they are sent. The default
:class:`~decent_bench.schemes.NoCompression` leaves messages unchanged.
:class:`~decent_bench.schemes.Quantization` rounds message entries to a fixed
number of significant digits. :class:`~decent_bench.schemes.TopK` and
:class:`~decent_bench.schemes.RandK` keep only a subset of coordinates and set
the rest to zero. :class:`~decent_bench.schemes.StochasticQuantization`
implements stochastic norm-scaled quantization used in QSGD.

Noise schemes perturb delivered messages. The default
:class:`~decent_bench.schemes.NoNoise` leaves messages unchanged, while
:class:`~decent_bench.schemes.GaussianNoise` adds independent Gaussian noise to
each message entry.

Drop schemes decide whether a message is lost before delivery. The default
:class:`~decent_bench.schemes.NoDrops` delivers every message.
:class:`~decent_bench.schemes.UniformDropRate` drops messages independently
with fixed probability, and :class:`~decent_bench.schemes.GilbertElliott`
models bursty drops with a two-state channel.

Message schemes can be provided as one shared instance or as a dictionary
mapping each sender agent to its own scheme. Use one instance per sender for
stateful schemes, since sharing a stateful scheme shares its internal state
across all senders.

.. code-block:: python

    from decent_bench.schemes import (
        GaussianNoise,
        StochasticQuantization,
        UniformActivationRate,
        UniformDropRate,
    )

    problem = BenchmarkProblem(
        network_structure=network_structure,
        costs=costs,
        x_optimal=x_optimal,
        agent_activations=[UniformActivationRate(0.8) for _ in costs],
        message_compression=StochasticQuantization(n_levels=8),
        message_noise=GaussianNoise(mean=0, std=0.001),
        message_drop=UniformDropRate(drop_rate=0.1),
    )