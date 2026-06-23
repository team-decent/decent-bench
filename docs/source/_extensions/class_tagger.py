"""
Sphinx extension that discovers tagged classes and renders them inline.

Usage in RST
------------
These are the available algorithms: :tagged:`gradient-based`.

Multiple tags for intersection:

These are the available algorithms:
:tagged:`gradient-based, peer-to-peer`

Configuration (conf.py)
-----------------------
tagged_list_modules = [
    "decent_bench.algorithms.p2p",
    "decent_bench.algorithms.federated",
    "decent_bench.costs",
]
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
import importlib
import inspect
from typing import TYPE_CHECKING

from docutils import nodes
from sphinx import addnodes
from sphinx.util.docutils import SphinxRole
from sphinx.util.logging import getLogger

from decent_bench.utils._tags import Tag

if TYPE_CHECKING:
    from sphinx.application import Sphinx


logger = getLogger(__name__)
_TAGS_ATTR = "_tags"
_CACHE: list["TaggedClass"] = []


@dataclass(frozen=True)
class TaggedClass:
    name: str
    module: str
    qualname: str
    tags: tuple[Tag, ...]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_tagged_classes(module_names: list[str]) -> list[TaggedClass]:
    """Import configured modules and collect tagged classes."""
    found: list[TaggedClass] = []
    seen: set[type] = set()

    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            logger.warning("Failed to import module %r: %s", module_name, exc)
            continue

        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls in seen:
                continue
            seen.add(cls)

            cls_tags = getattr(cls, _TAGS_ATTR, ())
            if not cls_tags:
                continue

            if not all(isinstance(tag, Tag) for tag in cls_tags):
                logger.warning(
                    "Skipping %s because %s contains non-Tag values",
                    f"{cls.__module__}.{cls.__qualname__}",
                    _TAGS_ATTR,
                )
                continue

            found.append(
                TaggedClass(
                    name=cls.__name__,
                    module=cls.__module__,
                    qualname=f"{cls.__module__}.{cls.__qualname__}",
                    tags=tuple(cls_tags),
                )
            )

    return found


def on_config_inited(app: Sphinx, config: object) -> None:
    """Populate the cache once per build."""
    module_names = getattr(config, "tagged_list_modules", [])
    _CACHE[:] = discover_tagged_classes(module_names)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_tag(value: str) -> Tag:
    """Parse one tag string into a Tag enum member."""
    try:
        return Tag(value)
    except ValueError as exc:
        allowed = ", ".join(tag.value for tag in Tag)
        raise ValueError(f"invalid tag {value!r}; allowed tags: {allowed}") from exc


def parse_tag_list(value: str) -> tuple[Tag, ...]:
    """Parse a comma-separated tag list."""
    raw_tags = [part.strip() for part in value.split(",") if part.strip()]
    if not raw_tags:
        raise ValueError("tag list must contain at least one tag")
    return tuple(parse_tag(raw) for raw in raw_tags)


def find_matching_classes(required_tags: tuple[Tag, ...]) -> list[TaggedClass]:
    """Return cached classes matching all requested tags."""
    return sorted(
        (
            cls
            for cls in _CACHE
            if all(tag in cls.tags for tag in required_tags)
        ),
        key=lambda cls: cls.name.lower(),
    )


def build_class_xref(cls: TaggedClass) -> addnodes.pending_xref:
    """Create a Sphinx cross-reference for one class."""
    xref = addnodes.pending_xref(
        "",
        refdomain="py",
        reftype="class",
        reftarget=f"{cls.module}.{cls.name}",
        refspecific=False,
    )
    xref += nodes.Text(cls.name)
    return xref


def build_inline_nodes(matches: list[TaggedClass]) -> list[nodes.Node]:
    """Build comma-separated inline nodes for matching classes."""
    result: list[nodes.Node] = []
    for i, cls in enumerate(matches):
        if i:
            result.append(nodes.Text(", "))
        result.append(build_class_xref(cls))
    return result


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

class TaggedRole(SphinxRole):
    """
    Inline role for tagged classes.

    Example:
        These are the available algorithms: :tagged:`gradient-based`
    """

    def run(self) -> tuple[list[nodes.Node], list[nodes.system_message]]:
        try:
            required_tags = parse_tag_list(self.text)
        except ValueError as exc:
            msg = self.inliner.reporter.error(str(exc), line=self.lineno)
            problem = self.inliner.problematic(self.rawtext, self.rawtext, msg)
            return [problem], [msg]

        matches = find_matching_classes(required_tags)

        if not matches:
            tag_text = ", ".join(tag.value for tag in required_tags)
            msg = self.inliner.reporter.warning(
                f"No classes found with tags [{tag_text}]",
                line=self.lineno,
            )
            return [nodes.inline(text="(none)")], [msg]

        return build_inline_nodes(matches), []


# ---------------------------------------------------------------------------
# Extension registration
# ---------------------------------------------------------------------------

def setup(app: Sphinx) -> dict[str, object]:
    """Register the extension with Sphinx."""
    app.add_config_value("tagged_list_modules", [], "env")
    app.connect("config-inited", on_config_inited)
    app.add_role("tagged", TaggedRole())

    return {
        "version": "0.1",
        "parallel_read_safe": False,
        "parallel_write_safe": False,
    }
