from typing import TYPE_CHECKING

from decent_bench.costs import EmpiricalRiskCost

if TYPE_CHECKING:
    from decent_bench.agents import Agent


def infer_client_data_size(client: "Agent") -> float:
    """
    Infer a client's local data size from an empirical-risk cost.

    Args:
        client: client agent whose data size should be inferred.

    Raises:
        ValueError: if the client cost is not an empirical-risk cost.

    """
    cost = client.cost
    if isinstance(cost, EmpiricalRiskCost):
        return float(cost.n_samples)

    raise ValueError(
        "Cannot infer client data size. Use an EmpiricalRiskCost to provide the number of local samples.",
    )
