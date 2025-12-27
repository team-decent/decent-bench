from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import matplotlib.axes
import networkx
import networkx as nx

_LAYOUT_FUNCS: dict[Literal["spring", "kamada_kawai", "circular", "random", "shell"], Any] = {
    "spring": nx.drawing.layout.spring_layout,
    "kamada_kawai": nx.drawing.layout.kamada_kawai_layout,
    "circular": nx.drawing.layout.circular_layout,
    "random": nx.drawing.layout.random_layout,
    "shell": nx.drawing.layout.shell_layout,
}


def plot_network(
    graph: networkx.Graph[Any],
    *,
    ax: matplotlib.axes.Axes | None = None,
    layout: Literal["spring", "kamada_kawai", "circular", "random", "shell"] = "spring",
    **draw_kwargs: Mapping[str, object],
) -> matplotlib.axes.Axes:
    """
    Plot a NetworkX graph using the built-in NetworkX drawing utilities.

    Args:
        graph: NetworkX graph to plot.
        ax: optional :class:`matplotlib.axes.Axes` to draw on. If ``None`` a new figure is created.
        layout: layout algorithm to position nodes (e.g. :func:`networkx.drawing.layout.spring_layout`,
            :func:`networkx.drawing.layout.kamada_kawai_layout`,
            :func:`networkx.drawing.layout.circular_layout`,
            :func:`networkx.drawing.layout.random_layout`,
            :func:`networkx.drawing.layout.shell_layout`).
        draw_kwargs: forwarded to :func:`networkx.drawing.nx_pylab.draw_networkx`.

    Returns:
        The matplotlib :class:`matplotlib.axes.Axes` containing the plot.

    Raises:
        RuntimeError: if matplotlib is not available.
        ValueError: if an unsupported layout is requested.

    """
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError("matplotlib is required for plotting the network") from exc

    layout_func = _LAYOUT_FUNCS.get(layout)
    if layout_func is None:
        supported = ", ".join(sorted(_LAYOUT_FUNCS))
        raise ValueError(f"Unsupported layout '{layout}'. Supported layouts: {supported}")

    pos = layout_func(graph)
    if ax is None:
        _, ax = plt.subplots()

    draw_kwargs_dict: dict[str, Any] = dict(draw_kwargs)
    nx.drawing.nx_pylab.draw_networkx(
        graph,
        pos=pos,
        ax=ax,
        **draw_kwargs_dict,
    )
    return ax
