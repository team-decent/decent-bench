"""
Utilities for algorithm initialization and general helpers.

This includes functions such as :func:`~decent_bench.algorithms.utils.infer_client_weight`,
which infers client data size from common cost attributes.
"""

import decent_bench.utils.interoperability as iop
from decent_bench.agents import Agent
from decent_bench.networks import FedNetwork, Network
from decent_bench.utils.array import Array
from decent_bench.utils.types import InitialStates


def initial_states(x0: InitialStates, network: Network) -> "dict[Agent, Array]":  # noqa: PLR0912
    """
    Build per-agent initial states, for use in :meth:`~decent_bench.algorithms.Algorithm.initialize`.

    Args:
        x0 (InitialStates):
            - ``None``: initialize all agents to zeros, using each agent's native shape/framework/device.
            - ``Array``: apply the same state to all agents.
            - ``dict[Agent, Array]``: explicit per-agent states.
        network (Network): network instance containing the target agents.

    Returns:
        dict[Agent, Array]: mapping from each network agent to its initial state.

    Raises:
        ValueError: if ``x0`` is missing required agent entries.
        TypeError: if ``x0`` has an invalid type.

    Notes:
        For :class:`~decent_bench.networks.FedNetwork`, explicit ``x0`` dictionaries must provide client entries.
        If the server entry is missing, it is inferred as the average of client initial states.
        Keys in ``x0`` not referring to agents in the network are silently ignored.

    """
    if x0 is None:
        x0s = {a: iop.zeros(a.cost.framework, a.cost.device, a.cost.shape) for a in network.graph}
    elif isinstance(x0, dict):
        # match by agent.id to handle deep-copied dicts whose keys are different instances
        x0_by_id = {}
        for a, v in x0.items():
            if not isinstance(a, Agent):
                raise TypeError(f"``x0`` must have keys of type Agent, got {type(a)}")
            if iop.shape(v) != a.cost.shape:
                raise ValueError(
                    f"Shape of initial state for agent {a} does not match cost shape: "
                    f"expected {a.cost.shape}, got {iop.shape(v)}"
                )
            x0_by_id[a.id] = v
        x0s = {}
        for a in network.agents():
            if a.id not in x0_by_id:
                raise ValueError(f"x0 not provided for agent {a}")
            x0s[a] = x0_by_id[a.id]
        if isinstance(network, FedNetwork):
            server = network.server()
            if server.id not in x0_by_id:
                x0s[server] = iop.mean(iop.stack([x0s[a] for a in network.clients()], dim=0), dim=0)
            else:
                x0s[server] = x0_by_id[server.id]
    elif iop.is_supported_array_type(x0):
        for a in network.graph:
            if iop.shape(x0) != a.cost.shape:
                raise ValueError(
                    f"Shape of initial state for agent {a} does not match cost shape: "
                    f"expected {a.cost.shape}, got {iop.shape(x0)}"
                )
        x0s = dict.fromkeys(network.graph, x0)
    else:
        raise ValueError(f"Invalid x0: expected None, Array, or dict[Agent, Array], got {type(x0)}")

    # ignore keys that are not network agents and normalize to the target framework/device
    return {a: iop.to_array(x0s[a], framework=a.cost.framework, device=a.cost.device) for a in network.graph}


def normal_initialization(
    network: Network,
    mean: float = 0.0,
    std: float = 1.0,
) -> "dict[Agent, Array]":
    """
    Build per-agent initial states sampled from a normal distribution.

    Args:
        network (Network): network instance containing the target agents.
        mean (float): mean of the normal distribution used to sample each state entry.
        std (float): standard deviation of the normal distribution used to sample each state entry.

    Returns:
        dict[Agent, Array]: mapping from each agent to an independently sampled random initial state.

    Notes:
        The states are created using each agent's own ``cost.shape``, ``cost.framework``, and ``cost.device``.

    """
    return {
        a: iop.normal(shape=a.cost.shape, framework=a.cost.framework, device=a.cost.device, mean=mean, std=std)
        for a in network.graph
    }


def uniform_initialization(
    network: Network,
    low: float = 0.0,
    high: float = 1.0,
) -> "dict[Agent, Array]":
    """
    Build per-agent initial states sampled from a uniform distribution.

    Args:
        network (Network): network instance containing the target agents.
        low (float): inclusive lower bound of the uniform distribution.
        high (float): exclusive upper bound of the uniform distribution.

    Returns:
        dict[Agent, Array]: mapping from each agent to an independently sampled random initial state.

    Raises:
        ValueError: if ``high`` is not greater than ``low``.

    Notes:
        The states are created using each agent's own ``cost.shape``, ``cost.framework``, and ``cost.device``.

    """
    if high <= low:
        raise ValueError(f"Expected high > low, got low={low} and high={high}.")

    return {
        a: iop.uniform(framework=a.cost.framework, device=a.cost.device, low=low, high=high, shape=a.cost.shape)
        for a in network.graph
    }


def pytorch_initialization(
    network: Network,
    all_same: bool = False,
) -> "dict[Agent, Array] | Array":
    """
    Build per-agent initial states using ``PyTorchCost.model`` initialization routine.

    Gets the initialized parameter tensor for every agent from
    ``PyTorchCost.model`` (via :meth:`torch.nn.Module.parameters`), and flattens it.
    The returned dict is compatible with :func:`initial_states` and can be passed
    directly as ``x0`` to any algorithm.

    Args:
        network (Network): network instance containing the target agents.
            All agents must have a :class:`~decent_bench.costs.PyTorchCost`.
        all_same (bool): if ``True``, use the first agent's initialized state for all agents.

    Returns:
        dict[Agent, Array] | Array: mapping from each network agent to its initial state,
        as a flattened parameter vector extracted from the initialized model.

    Raises:
        TypeError: if any agent's cost is not a :class:`~decent_bench.costs.PyTorchCost`.

    """
    from decent_bench.costs import PyTorchCost  # noqa: PLC0415

    if all_same:
        # use the first agent's initialized model as the shared initial state for all agents
        first_agent = next(iter(network.agents()))
        if not isinstance(first_agent.cost, PyTorchCost):
            raise TypeError(
                f"Agent {first_agent} has cost of type {type(first_agent.cost).__name__!r}, expected PyTorchCost."
            )
        return iop.to_array(
            first_agent.cost._get_model_parameters(),  # noqa: SLF001
            framework=first_agent.cost.framework,
            device=first_agent.cost.device,
        )

    x0s = {}
    for a in network.graph:
        if not isinstance(a.cost, PyTorchCost):
            raise TypeError(f"Agent {a} has cost of type {type(a.cost).__name__!r}, expected PyTorchCost.")
        val = iop.to_array(
            a.cost._get_model_parameters(),  # noqa: SLF001
            framework=a.cost.framework,
            device=a.cost.device,
        )
        x0s[a] = val
    return x0s


def infer_client_weight(client: "Agent") -> float:
    """
    Infer a client's weight from its cost data size.

    Prefers an explicit ``n_samples`` attribute when present, then falls back to common size-bearing attributes such
    as ``A`` or ``b`` on the cost.

    Raises:
        ValueError: if a suitable positive size attribute is not found on the cost.

    """

    def _validate_size(size: float) -> float:
        if size <= 0:
            raise ValueError("Client data size must be positive")
        return size

    cost = client.cost
    if hasattr(cost, "n_samples"):
        n_samples = cost.n_samples
        if n_samples is not None:
            return _validate_size(float(n_samples))
    if hasattr(cost, "A"):
        try:
            size = iop.shape(cost.A)[0]
        except Exception:
            size = None
        if size is not None:
            return _validate_size(float(size))
    if hasattr(cost, "b"):
        try:
            size = iop.shape(cost.b)[0]
        except Exception:
            size = None
        if size is not None:
            return _validate_size(float(size))
    raise ValueError(
        "Cannot infer client data size. Add a size attribute to the cost or use uniform aggregation instead."
    )
