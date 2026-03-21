"""Utilities for tagging classes, to build tag-based lists in the docs."""

from collections.abc import Callable


def tags(*tags: str) -> Callable[[type], type]:
    """
    Decorate any class with tags.

    Tags are stored on the class as the ``_tags`` attribute and are used
    by the Sphinx ``class_tagger`` extension to build per-tag lists in the docs.

    Args:
        *tags: One or more tag strings (e.g. ``"gradient-based"``, ``"federated"``).

    Returns:
        A class decorator that attaches the tags to the decorated class.

    Example:
        .. code-block:: python

            from decent_bench.utils.tags import tags

            @tags("gradient-based", "peer-to-peer")
            class DGD(P2PAlgorithm):
                ...

    """

    def decorator(cls: type) -> type:
        setattr(cls, "_tags", tuple(tags))
        return cls

    return decorator
