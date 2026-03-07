from typing import TYPE_CHECKING

import decent_bench.utils.interoperability as iop
from decent_bench.networks import Network
from decent_bench.utils.array import Array

if TYPE_CHECKING:
    from decent_bench.agents import Agent


def zero_initialization(x0: Array | None, network: Network, stacked_copies: int | None = None) -> Array:
    """
    Initialize the variable to zero if x0 is None.

    If stacked_copies is provided, stack the variable ``stacked_copies`` times.

    Args:
        x0 (Array | None): optional initial variable
        network (Network): network instance
        stacked_copies (int | None): how many times to stack the variable

    Returns:
        initialized variable

    Raises:
        ValueError: if the shape of x0 does not match the expected shape

    """
    i = network.agents()[0]
    if x0 is None:
        x0 = iop.zeros(framework=i.cost.framework, shape=i.cost.shape, device=i.cost.device)

    if iop.shape(x0) != i.cost.shape:
        raise ValueError(f"Initial variable has shape {iop.shape(x0)}, expected {i.cost.shape}.")

    if stacked_copies is not None:
        x0 = iop.stack([x0 for _ in range(stacked_copies)])

    return iop.to_array(x0, framework=i.cost.framework, device=i.cost.device)


def randn_initialization(
    x0: Array | None,
    network: Network,
    stacked_copies: int | None = None,
    mean: float = 0.0,
    std: float = 1.0,
) -> Array:
    """
    Initialize the variable to normally distributed random values if x0 is None.

    If stacked_copies is provided, stack the variable ``stacked_copies`` times.

    Args:
        x0 (Array | None): optional initial variable
        network (Network): network instance
        stacked_copies (int | None): how many times to stack the variable
        mean (float): mean for random values
        std (float): standard deviation for random values

    Returns:
        initialized variable

    Raises:
        ValueError: if the shape of x0 does not match the expected shape

    """
    i = network.agents()[0]
    if x0 is None:
        x0 = iop.randn(framework=i.cost.framework, shape=i.cost.shape, device=i.cost.device, mean=mean, std=std)

    if iop.shape(x0) != i.cost.shape:
        raise ValueError(f"Initial variable has shape {iop.shape(x0)}, expected {i.cost.shape}.")

    if stacked_copies is not None:
        x0 = iop.stack([x0 for _ in range(stacked_copies)])

    return iop.to_array(x0, framework=i.cost.framework, device=i.cost.device)


def infer_client_weight(client: "Agent") -> float:
    """
    Infer a client's weight from its cost data size.

    Looks for common attributes such as ``A``, ``b``, or ``n_samples`` on the cost.

    Raises:
        ValueError: if a suitable size attribute is not found on the cost.

    """
    cost = client.cost
    if hasattr(cost, "A"):
        try:
            size = iop.shape(cost.A)[0]
        except Exception:
            size = None
        if size is not None:
            return float(size)
    if hasattr(cost, "b"):
        try:
            size = iop.shape(cost.b)[0]
        except Exception:
            size = None
        if size is not None:
            return float(size)
    if hasattr(cost, "n_samples"):
        n_samples = cost.n_samples
        if n_samples is not None:
            return float(n_samples)
    raise ValueError(
        "Cannot infer client data size. Provide client_weights to the algorithm or add a size attribute to the cost."
    )
