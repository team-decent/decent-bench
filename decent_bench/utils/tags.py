"""Utilities for tagging algorithm classes with category labels."""

from collections.abc import Callable


def algorithm_tags(*tags: str) -> Callable[[type], type]:
    """
    Decorate an :class:`~decent_bench.distributed_algorithms.Algorithm` subclass with category tags.

    Tags are stored on the class as the ``_algorithm_tags`` attribute and are used
    by the Sphinx ``algorithm_tagger`` extension to build per-tag index pages.

    Args:
        *tags: One or more tag strings (e.g. ``"gradient-based"``, ``"federated"``).

    Returns:
        A class decorator that attaches the tags to the decorated class.

    Example:
        .. code-block:: python

            from decent_bench.utils.tags import algorithm_tags

            @algorithm_tags("gradient-based", "decentralized")
            class DGD(P2PAlgorithm):
                ...

    """

    def decorator(cls: type) -> type:
        cls._algorithm_tags = tuple(tags)
        return cls

    return decorator
