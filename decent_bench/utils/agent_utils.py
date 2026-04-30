from typing import TYPE_CHECKING, Any

import decent_bench.utils.interoperability as iop

if TYPE_CHECKING:
    from decent_bench.agents import Agent


def _validate_positive_client_data_size(size: Any) -> float:  # noqa: ANN401
    try:
        size_float = float(size)
    except (TypeError, ValueError) as exc:
        raise ValueError("Client data size must be numeric") from exc
    if size_float <= 0:
        raise ValueError("Client data size must be positive")
    return size_float


def infer_client_data_size(client: "Agent", data_size_key: str = "n_samples") -> float:
    """
    Infer a client's local data size from agent metadata or common cost attributes.

    The lookup order is:

    - ``client.data[data_size_key]``
    - ``client.cost.n_samples``
    - the first dimension of ``client.cost.A``
    - the first dimension of ``client.cost.b``

    Args:
        client: client agent whose data size should be inferred.
        data_size_key: key used to read explicit data sizes from ``client.data``.

    Raises:
        ValueError: if no positive data-size signal can be found.

    """
    if data_size_key in client.data:
        return _validate_positive_client_data_size(client.data[data_size_key])

    cost = client.cost
    if hasattr(cost, "n_samples"):
        n_samples = cost.n_samples
        if n_samples is not None:
            return _validate_positive_client_data_size(n_samples)
    if hasattr(cost, "A"):
        try:
            size = iop.shape(cost.A)[0]
        except Exception:
            size = None
        if size is not None:
            return _validate_positive_client_data_size(size)
    if hasattr(cost, "b"):
        try:
            size = iop.shape(cost.b)[0]
        except Exception:
            size = None
        if size is not None:
            return _validate_positive_client_data_size(size)
    raise ValueError(
        "Cannot infer client data size. Add client data metadata or a size attribute to the cost.",
    )
