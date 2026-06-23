"""Utilities for tagging classes, to build tag-based lists in the docs."""

from collections.abc import Callable
from enum import StrEnum


class Tag(StrEnum):
    # types of classes
    ALGORITHM = "algorithm"
    COST = "cost"
    METRIC = "metric"
    RUNTIME_METRIC = "runtime metric"

    # architecture
    FEDERATED = "federated"
    PEER_TO_PEER = "peer-to-peer"

    # algorithm types
    GRADIENT_BASED = "gradient-based"
    GRADIENT_TRACKING = "gradient tracking"
    ADMM = "ADMM"
    DUAL_METHOD = "dual method"
    VARIANCE_REDUCTION = "variance reduction"

    # cost types
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    EMPIRICAL_RISK = "empirical-risk"
    REGULARIZER = "regularizer"


def tags[T: type](*tags: Tag) -> Callable[[T], T]:
    """
    Decorate any class with tags.

    Tags are stored on the class as the ``_tags`` attribute and are used
    by the Sphinx ``class_tagger`` extension to build per-tag lists in the docs.

    Args:
        *tags: One or more tags from enum ``Tag``.

    Returns:
        A class decorator that attaches the tags to the decorated class.

    Example:
        .. code-block:: python

            from decent_bench.utils._tags import tags, Tag

            @tags(Tag.PEER_TO_PEER, Tag.GRADIENT_BASED)
            class DGD(P2PAlgorithm):
                ...

    """
    if not tags:
        raise ValueError("at least one tag is required")
    if not all(isinstance(tag, Tag) for tag in tags):
        raise TypeError("tags() accepts Tag enum members only")

    def decorator(cls: T) -> T:
        cls._tags = tuple(tags)  # type: ignore[attr-defined]
        return cls

    return decorator
