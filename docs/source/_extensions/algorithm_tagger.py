"""Sphinx extension that discovers tagged algorithm classes and renders tag-based index pages.

Usage in RST
------------
List algorithms that carry a specific tag::

    .. algorithm-list::
       :tag: gradient-based

Generate a full index (one section per tag)::

    .. algorithm-index::
       :tag: federated

Both directives scan all subclasses of
:class:`~decent_bench.distributed_algorithms.Algorithm` and
:class:`~decent_bench.centralized_algorithms.Algorithm` (when present) that
have been decorated with
:func:`~decent_bench.utils.tags.algorithm_tags`.
"""

from __future__ import annotations

import importlib
import inspect
from typing import TYPE_CHECKING, Any, ClassVar

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.util.docutils import SphinxDirective

if TYPE_CHECKING:
    from sphinx.application import Sphinx

# ---------------------------------------------------------------------------
# Algorithm discovery helpers
# ---------------------------------------------------------------------------

_ALGORITHM_MODULES = [
    "decent_bench.distributed_algorithms",
    "decent_bench.centralized_algorithms",
]


def _discover_algorithms() -> list[dict[str, Any]]:
    """Return a list of metadata dicts for every tagged Algorithm subclass.

    Each dict has the keys ``name``, ``qualname``, ``module``, ``tags``, and
    ``summary`` (first line of the docstring, or an empty string).
    """
    results: list[dict[str, Any]] = []
    seen: set[type] = set()

    for module_name in _ALGORITHM_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue

        for _attr_name, obj in inspect.getmembers(module, inspect.isclass):
            if obj in seen:
                continue
            seen.add(obj)

            tags: tuple[str, ...] = getattr(obj, "_algorithm_tags", ())
            if not tags:
                continue

            docstring = inspect.getdoc(obj) or ""
            summary = docstring.splitlines()[0] if docstring else ""

            results.append(
                {
                    "name": obj.__name__,
                    "qualname": f"{obj.__module__}.{obj.__qualname__}",
                    "module": obj.__module__,
                    "tags": tags,
                    "summary": summary,
                }
            )

    return results


# ---------------------------------------------------------------------------
# Cached lookup (populated once per Sphinx build via config-inited)
# ---------------------------------------------------------------------------

_ALGORITHM_CACHE: list[dict[str, Any]] = []


def _cache_algorithms(app: Sphinx, _config: Any) -> None:  # noqa: ANN401
    """Discover and cache tagged algorithms in the module-level cache."""
    _ALGORITHM_CACHE.clear()
    _ALGORITHM_CACHE.extend(_discover_algorithms())


# ---------------------------------------------------------------------------
# Directive nodes
# ---------------------------------------------------------------------------


class algorithm_list_node(nodes.General, nodes.Element):  # noqa: N801
    """Placeholder node for the ``algorithm-list`` directive."""


class algorithm_index_node(nodes.General, nodes.Element):  # noqa: N801
    """Placeholder node for the ``algorithm-index`` directive."""


# ---------------------------------------------------------------------------
# Directives
# ---------------------------------------------------------------------------


class AlgorithmListDirective(SphinxDirective):
    """Render a bullet list of algorithms filtered by tag.

    Options
    -------
    tag : str
        Only algorithms decorated with this tag are shown.

    Example::

        .. algorithm-list::
           :tag: gradient-based
    """

    has_content = False
    required_arguments = 0
    optional_arguments = 0
    option_spec: ClassVar[dict[str, Any]] = {
        "tag": directives.unchanged_required,
    }

    def run(self) -> list[nodes.Node]:
        tag = self.options["tag"]
        node = algorithm_list_node()
        node["tag"] = tag
        return [node]


class AlgorithmIndexDirective(SphinxDirective):
    """Render a full index of algorithms, optionally filtered to one tag.

    Options
    -------
    tag : str, optional
        When provided only algorithms with this tag are included; otherwise
        all tagged algorithms are shown, grouped by tag.

    Example::

        .. algorithm-index::

        .. algorithm-index::
           :tag: federated
    """

    has_content = False
    required_arguments = 0
    optional_arguments = 0
    option_spec: ClassVar[dict[str, Any]] = {
        "tag": directives.unchanged,
    }

    def run(self) -> list[nodes.Node]:
        node = algorithm_index_node()
        node["tag"] = self.options.get("tag", "")
        return [node]


# ---------------------------------------------------------------------------
# Node visitors (resolve placeholder nodes into real docutils nodes)
# ---------------------------------------------------------------------------


def _make_algorithm_entry(alg: dict[str, Any]) -> nodes.list_item:
    """Return a list item for a single algorithm."""
    para = nodes.paragraph()

    # Cross-reference to the class API docs
    ref = nodes.reference(
        "",
        alg["name"],
        internal=False,
        refuri=f"api/{alg['module']}.html#{alg['qualname']}",
    )
    para += ref

    if alg["summary"]:
        para += nodes.Text(f" — {alg['summary']}")

    tag_inline = nodes.inline()
    tag_inline += nodes.Text(" [")
    for i, tag in enumerate(alg["tags"]):
        if i:
            tag_inline += nodes.Text(", ")
        tag_inline += nodes.literal(text=tag)
    tag_inline += nodes.Text("]")
    para += tag_inline

    item = nodes.list_item()
    item += para
    return item


def visit_algorithm_list_node(self: Any, node: algorithm_list_node) -> None:  # noqa: ANN401
    """Resolve an :class:`algorithm_list_node` into a bullet list."""
    tag: str = node["tag"]

    matching = [a for a in _ALGORITHM_CACHE if tag in a["tags"]]

    if not matching:
        warning = nodes.warning()
        warning += nodes.paragraph(text=f"No algorithms found with tag '{tag}'.")
        node.replace_self(warning)
        return

    bullet_list = nodes.bullet_list()
    for alg in matching:
        bullet_list += _make_algorithm_entry(alg)

    node.replace_self(bullet_list)


def depart_algorithm_list_node(_self: Any, _node: algorithm_list_node) -> None:
    """No closing action needed."""


def visit_algorithm_index_node(self: Any, node: algorithm_index_node) -> None:  # noqa: ANN401
    """Resolve an :class:`algorithm_index_node` into grouped sections."""
    filter_tag: str = node["tag"]

    if filter_tag:
        matching = [a for a in _ALGORITHM_CACHE if filter_tag in a["tags"]]
        groups: dict[str, list[dict[str, Any]]] = {filter_tag: matching} if matching else {}
    else:
        groups = {}
        for alg in _ALGORITHM_CACHE:
            for t in alg["tags"]:
                groups.setdefault(t, []).append(alg)

    if not groups:
        label = f"'{filter_tag}'" if filter_tag else "any tag"
        warning = nodes.warning()
        warning += nodes.paragraph(text=f"No algorithms found with {label}.")
        node.replace_self(warning)
        return

    container = nodes.container()
    for tag_name in sorted(groups):
        section = nodes.section(ids=[f"tag-{tag_name}"])
        section += nodes.title(text=tag_name)
        bullet_list = nodes.bullet_list()
        for alg in groups[tag_name]:
            bullet_list += _make_algorithm_entry(alg)
        section += bullet_list
        container += section

    node.replace_self(container)


def depart_algorithm_index_node(_self: Any, _node: algorithm_index_node) -> None:
    """No closing action needed."""


# ---------------------------------------------------------------------------
# Extension setup
# ---------------------------------------------------------------------------


def setup(app: Sphinx) -> dict[str, Any]:
    """Register the extension with Sphinx."""
    app.connect("config-inited", _cache_algorithms)

    app.add_node(
        algorithm_list_node,
        html=(visit_algorithm_list_node, depart_algorithm_list_node),
        latex=(visit_algorithm_list_node, depart_algorithm_list_node),
        text=(visit_algorithm_list_node, depart_algorithm_list_node),
    )
    app.add_node(
        algorithm_index_node,
        html=(visit_algorithm_index_node, depart_algorithm_index_node),
        latex=(visit_algorithm_index_node, depart_algorithm_index_node),
        text=(visit_algorithm_index_node, depart_algorithm_index_node),
    )

    app.add_directive("algorithm-list", AlgorithmListDirective)
    app.add_directive("algorithm-index", AlgorithmIndexDirective)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
