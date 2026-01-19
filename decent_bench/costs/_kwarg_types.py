from typing import NotRequired, TypedDict


class CostKwargs(TypedDict):
    """Keyword arguments for cost functions."""

    # Append more keyword arguments as needed.
    indices: NotRequired[list[int]]
