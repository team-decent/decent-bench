from collections.abc import Sequence

import decent_bench.utils.interoperability as iop
from decent_bench.costs import Cost, QuadraticCost
from decent_bench.utils.array import Array
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks


def create_consensus_problem(
    size: int = 10,
    n_agents: int = 100,
) -> tuple[Sequence[Cost], Array, Sequence[Array]]:
    """
    Create consensus problems.

    Args:
        size: number of dimensions
        n_agents: number of agents

    """
    u = [iop.normal(shape=(size,), std=10, framework=SupportedFrameworks.NUMPY, device=SupportedDevices.CPU) for _ in range(n_agents)]

    costs = [QuadraticCost(iop.eye_like(u[i]), -u[i]) for i in range(n_agents)]
    x_optimal = iop.mean(iop.stack(u), dim=0)

    return costs, x_optimal, u
