from __future__ import annotations

from dataclasses import dataclass

import importlib
import inspect
from typing import TYPE_CHECKING, ClassVar

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx import addnodes
from sphinx.util.docutils import SphinxDirective
from sphinx.util.logging import getLogger

from decent_bench.utils._tags import Tag

if TYPE_CHECKING:
    from sphinx.application import Sphinx


logger = getLogger(__name__)
_TAGS_ATTR = "_tags"
_CACHE: list["TaggedClass"] = []


# ---------------------------------------------------------------------------
# Phase 0: Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaggedClass:
    name: str
    qualname: str
    module: str
    tags: tuple[Tag, ...]


# ---------------------------------------------------------------------------
# Phase 1: Build-time discovery
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

            tags_ = getattr(cls, _TAGS_ATTR, ())
            if not tags_:
                continue
            if not all(isinstance(tag, Tag) for tag in tags_):
                logger.warning("Class %s has non-Tag values in %s", cls, _TAGS_ATTR)
                continue

            found.append(
                TaggedClass(
                    name=cls.__name__,
                    qualname=f"{cls.__module__}.{cls.__qualname__}",
                    module=cls.__module__,
                    tags=tuple(tags_),
                )
            )

    return found


def on_config_inited(app: Sphinx, config: object) -> None:
    """Populate the cache once per build."""
    _CACHE[:] = discover_tagged_classes(getattr(config, "tagged_list_modules", []))


# ---------------------------------------------------------------------------
# Phase 2: Directive parsing and rendering
# ---------------------------------------------------------------------------

def parse_tag_list(value: str) -> tuple[Tag, ...]:
    """Parse a comma-separated tag list into Tag members."""
    raw_tags = [part.strip() for part in value.split(",") if part.strip()]
    if not raw_tags:
        raise ValueError("tag option must contain at least one tag")

    try:
        return tuple(Tag(raw) for raw in raw_tags)
    except ValueError as exc:
        allowed = ", ".join(tag.value for tag in Tag)
        raise ValueError(f"invalid tag; allowed tags: {allowed}") from exc


class TaggedListDirective(SphinxDirective):
    """Render tagged classes as comma-separated class cross-references."""

    has_content = False
    option_spec: ClassVar[dict[str, object]] = {
        "tag": parse_tag_list,
        "module": directives.unchanged,
    }

    def run(self) -> list[nodes.Node]:
        required_tags: tuple[Tag, ...] = self.options["tag"]
        module_prefix: str = self.options.get("module", "")

        matches = sorted(
            (
                cls
                for cls in _CACHE
                if all(tag in cls.tags for tag in required_tags)
                and (not module_prefix or cls.module.startswith(module_prefix))
            ),
            key=lambda cls: cls.name.lower(),
        )

        if not matches:
            tag_text = ", ".join(tag.value for tag in required_tags)
            detail = f" in module '{module_prefix}'" if module_prefix else ""
            return [nodes.paragraph(text=f"No classes found with tags [{tag_text}]{detail}.")]

        paragraph = nodes.paragraph()
        for i, cls in enumerate(matches):
            if i:
                paragraph += nodes.Text(", ")

            xref = addnodes.pending_xref(
                "",
                refdomain="py",
                reftype="class",
                reftarget=f"{cls.module}.{cls.name}",
                refspecific=False,
            )
            xref += nodes.Text(cls.name)
            paragraph += xref

        return [paragraph]


# ---------------------------------------------------------------------------
# Phase 3: Extension registration
# ---------------------------------------------------------------------------

def setup(app: Sphinx) -> dict[str, object]:
    app.add_config_value("tagged_list_modules", [], "env")
    app.connect("config-inited", on_config_inited)
    app.add_directive("tagged-list", TaggedListDirective)

    return {
        "version": "0.5",
        "parallel_read_safe": False,
        "parallel_write_safe": False,
    }