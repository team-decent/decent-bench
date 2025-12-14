import decent_bench.utils.interoperability as iop
from decent_bench.networks import P2PNetwork
from decent_bench.utils.array import Array


def zero_initialization(x0: Array | None, network: P2PNetwork, stack_to: int | None = None) -> Array:
    """
    Initialize the variable to zero if x0 is None.

    If stack_to is provided, stack the variable ``stack_to`` times.

    Args:
        x0 (Array | None): optional initial variable
        network (P2PNetwork): peer-to-peer network
        stack_to (int | None): how many times to stack the variable

    Returns:
        initialized variable

    """
    i = network.agents()[0]
    if x0 is None:
        x0 = iop.zeros(framework=i.cost.framework, shape=i.cost.shape, device=i.cost.device)

    if stack_to is not None:
        x0 = iop.stack([x0 for _ in range(stack_to)])

    return iop.to_array(x0, framework=i.cost.framework, device=i.cost.device)


def randn_initialization(
    x0: Array | None,
    network: P2PNetwork,
    stack_to: int | None = None,
    mean: float = 0.0,
    std: float = 1.0,
) -> Array:
    """
    Initialize the variable to normally distributed random values if x0 is None.

    If stack_to is provided, stack the variable ``stack_to`` times.

    Args:
        x0 (Array | None): optional initial variable
        network (P2PNetwork): peer-to-peer network
        stack_to (int | None): how many times to stack the variable
        mean (float): mean for random values
        std (float): standard deviation for random values

    Returns:
        initialized variable

    """
    i = network.agents()[0]
    if x0 is None:
        x0 = iop.randn(framework=i.cost.framework, shape=i.cost.shape, device=i.cost.device, mean=mean, std=std)

    if stack_to is not None:
        x0 = iop.stack([x0 for _ in range(stack_to)])

    return iop.to_array(x0, framework=i.cost.framework, device=i.cost.device)
