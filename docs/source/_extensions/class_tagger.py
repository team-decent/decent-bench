"""Sphinx extension that discovers tagged classes and renders tag-based lists.

Usage in RST
------------
List classes that carry a specific tag::

    .. tagged-list::
       :tag: gradient-based

Scans all modules listed in ``tagged_list_modules`` in ``conf.py`` for classes
decorated with :func:`~decent_bench.utils.tags.tags`.

Configuration (conf.py)
-----------------------
Add the modules to scan::

    tagged_list_modules = [
        "decent_bench.distributed_algorithms",
        "decent_bench.costs",
    ]
"""

from __future__ import annotations

import importlib
import inspect
from typing import TYPE_CHECKING, Any, ClassVar

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.util.docutils import SphinxDirective
from sphinx.util.logging import getLogger

if TYPE_CHECKING:
    from sphinx.application import Sphinx

# ---------------------------------------------------------------------------
# Class discovery helpers
# ---------------------------------------------------------------------------

_TAGS_ATTR = "_tags"
logger = getLogger(__name__)


def _discover_tagged_classes(module_names: list[str]) -> list[dict[str, Any]]:
    """Return a list of metadata dicts for every tagged class in ``module_names``.

    Each dict has the keys ``name``, ``qualname``, ``module``, and ``tags``.
    """
    results: list[dict[str, Any]] = []
    seen: set[type] = set()

    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            logger.warning(f"Failed to import module '{module_name}': {e}")
            continue

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj in seen:
                continue
            seen.add(obj)

            tags: tuple[str, ...] = getattr(obj, _TAGS_ATTR, ())
            if not tags:
                continue

            results.append(
                {
                    "name": obj.__name__,
                    "qualname": f"{obj.__module__}.{obj.__qualname__}",
                    "module": obj.__module__,
                    "tags": tags,
                }
            )

    return results


# ---------------------------------------------------------------------------
# Cached lookup (populated once per Sphinx build via config-inited)
# ---------------------------------------------------------------------------

_CLASS_CACHE: list[dict[str, Any]] = []


def _cache_tagged_classes(app: Sphinx, config: Any) -> None:  # noqa: ANN401
    """Discover and cache tagged classes in the module-level cache."""
    module_names: list[str] = getattr(config, "tagged_list_modules", [])
    _CLASS_CACHE.clear()
    _CLASS_CACHE.extend(_discover_tagged_classes(module_names))


# ---------------------------------------------------------------------------
# Directive nodes
# ---------------------------------------------------------------------------


class tagged_list_node(nodes.General, nodes.Element):  # noqa: N801
    """Placeholder node for the ``tagged-list`` directive."""


# ---------------------------------------------------------------------------
# Directives
# ---------------------------------------------------------------------------


class TaggedListDirective(SphinxDirective):
    """Render a bullet list of classes filtered by tag.

    Options
    -------
    tag : str
        Only classes decorated with this tag are shown.
    module : str, optional
        When provided, only classes whose ``__module__`` starts with this
        value are included (e.g. ``decent_bench.costs``).

    Example::

        .. tagged-list::
           :tag: gradient-based

        .. tagged-list::
           :tag: regression
           :module: decent_bench.costs
    """

    has_content = False
    required_arguments = 0
    optional_arguments = 0
    option_spec: ClassVar[dict[str, Any]] = {
        "tag": directives.unchanged_required,
        "module": directives.unchanged,
    }

    def run(self) -> list[nodes.Node]:
        node = tagged_list_node()
        node["tag"] = self.options["tag"]
        node["module"] = self.options.get("module", "")
        return [node]


# ---------------------------------------------------------------------------
# Node resolution
# ---------------------------------------------------------------------------


def _make_entry(cls: dict[str, Any], exclude_tag: str = "") -> nodes.list_item:
    """Return a list item for a single tagged class."""
    para = nodes.paragraph()

    ref = nodes.reference(
        "",
        cls["name"],
        internal=False,
        refuri=f"api/{cls['module']}.html#{cls['qualname']}",
    )
    para += ref

    remaining_tags = [t for t in cls["tags"] if t != exclude_tag]
    if remaining_tags:
        tag_inline = nodes.inline()
        tag_inline += nodes.Text(" [")
        for i, tag in enumerate(remaining_tags):
            if i:
                tag_inline += nodes.Text(", ")
            tag_inline += nodes.literal(text=tag)
        tag_inline += nodes.Text("]")
        para += tag_inline

    item = nodes.list_item()
    item += para
    return item


def _resolve_tagged_nodes(app: Sphinx, doctree: nodes.document, docname: str) -> None:
    """Replace placeholder nodes with rendered content in the doctree-resolved phase."""
    for node in doctree.findall(tagged_list_node):
        tag: str = node["tag"]
        module_filter: str = node["module"]

        matching = [
            c for c in _CLASS_CACHE
            if tag in c["tags"]
            and (not module_filter or c["module"].startswith(module_filter))
        ]

        if not matching:
            detail = f" in module '{module_filter}'" if module_filter else ""
            warning = nodes.warning()
            warning += nodes.paragraph(text=f"No classes found with tag '{tag}'{detail}.")
            node.replace_self(warning)
            continue

        bullet_list = nodes.bullet_list()
        for cls in matching:
            bullet_list += _make_entry(cls, exclude_tag=tag)
        node.replace_self(bullet_list)


def _visit_passthrough(self: Any, node: nodes.Node) -> None:  # noqa: ANN401
    """Fallback visitor — should not be reached after doctree-resolved replacement."""


def _depart_passthrough(_self: Any, _node: nodes.Node) -> None:
    """No closing action needed."""


# ---------------------------------------------------------------------------
# Extension setup
# ---------------------------------------------------------------------------


def setup(app: Sphinx) -> dict[str, Any]:
    """Register the extension with Sphinx."""
    app.add_config_value("tagged_list_modules", default=[], rebuild="env")
    app.connect("config-inited", _cache_tagged_classes)
    app.connect("doctree-resolved", _resolve_tagged_nodes)

    app.add_node(
        tagged_list_node,
        html=(_visit_passthrough, _depart_passthrough),
        latex=(_visit_passthrough, _depart_passthrough),
        text=(_visit_passthrough, _depart_passthrough),
    )
    app.add_directive("tagged-list", TaggedListDirective)

    return {
        "version": "0.1",
        "parallel_read_safe": False,
        "parallel_write_safe": False,
    }
