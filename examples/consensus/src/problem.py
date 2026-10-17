from collections.abc import Sequence

from decent_array import Array
from decent_array import interoperability as iop

from decent_bench.costs import Cost, QuadraticCost


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
    u = [iop.normal(shape=(size,), std=10) for _ in range(n_agents)]

    costs = [QuadraticCost(iop.eye(size), -u[i]) for i in range(n_agents)]
    x_optimal = iop.mean(iop.stack(u), axis=0)

    return costs, x_optimal, u
