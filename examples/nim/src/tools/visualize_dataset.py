import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import NDArray
from PIL import Image
from src.dataset import NIMDatasetHandler
from src.lidar import (
    RayCastHit,
    compute_headings,
    densify_path,
    image_to_occupancy,
    simulate_lidar_scan,
)

from decent_bench.utils.array import Array


def _make_red_occupancy_overlay(
    occ_bool: NDArray[np.bool_],
    *,
    alpha: float = 0.4,
) -> NDArray[np.float64]:
    """Build an RGBA image with pure red occupancy pixels and transparent background."""
    overlay = np.zeros((*occ_bool.shape, 4), dtype=np.float64)
    overlay[..., 0] = 1.0  # Red channel
    overlay[..., 3] = occ_bool.astype(np.float64) * alpha  # Per-pixel alpha
    return overlay


def visualize_nim_dataset(
    nim_data: NIMDatasetHandler,
    *,
    figsize: tuple[int, int] = (12, 10),
    animate: bool = False,
    path_index: int | list[int] | None = None,
    fps: int = 10,
    save_path: str | None = None,
) -> FuncAnimation | Figure:
    """
    Visualizes a NIMDatasetHandler instance.

    If `nim_data.paths` is not set, a static visualization of the spatial partitions
    and sampled points is produced. If `nim_data.paths` is set, the function visualizes the paths and simulated
    lidar hits. In this case, if `animate` is False, a static plot with all paths and hits is shown. If
    `animate` is True, an animation of the scanner moving along the specified path(s) and shooting beams is created.

    Args:
        nim_data: A NIMDatasetHandler instance to visualize.
        figsize: The size of the figure for static visualizations.
        animate: Whether to create an animation (only applicable if nim_data.paths is set).
        path_index: The index (or list of indices) of the path(s) to visualize. If None, all paths are shown.
        fps: Frames per second for the animation (if animate=True).
        save_path: If provided, the visualization (static or animated) will be saved to this path.

    Returns:
        A matplotlib Figure for static visualizations, or a FuncAnimation for animated visualizations.

    Note:
        Remember to call matplotlib.pyplot.show() to display the visualization if not
        running in an interactive environment.

    """
    if nim_data.paths is None:
        return _visualize_random_sampling(
            nim_data,
            figsize=figsize,
            save_path=save_path,
        )
    return _visualize_lidar_sampling(
        nim_data,
        path_index=path_index,
        animate=animate,
        fps=fps,
        figsize=figsize,
        save_path=save_path,
    )


def _visualize_random_sampling(
    nim_data: NIMDatasetHandler,
    figsize: tuple[int, int] = (12, 10),
    save_path: str | None = None,
) -> Figure:
    """
    Visualize the spatial partitions and sampled points from a NIMData dataset.

    Args:
        nim_data: A NIMData instance
        figsize: The figure size
        save_path: If given, the figure is saved to this path.

    Returns:
        The matplotlib Figure object containing the visualization.

    """
    # Load and display the original image
    image = Image.open(nim_data.image_file).convert("L")

    # Create figure and display image
    fig, ax = plt.subplots(figsize=figsize)
    img_array = np.array(image)
    ax.imshow(img_array, cmap="gray", origin="upper")
    ax.axis("off")

    # Overlay occupied pixels in red for easier map interpretation.
    img_float = img_array.astype(np.float64) / 255.0
    occ = image_to_occupancy(img_float, threshold=nim_data.occupancy_threshold)
    occ_bool = occ.astype(bool)
    occ_overlay = _make_red_occupancy_overlay(occ_bool, alpha=0.4)
    ax.imshow(
        occ_overlay,
        origin="upper",
        interpolation="nearest",
    )

    # Calculate grid dimensions using the same algorithm as _create_spatial_partitions
    rows = int(np.floor(np.sqrt(nim_data.n_partitions)))
    while nim_data.n_partitions % rows != 0 and rows > 1:
        rows -= 1
    cols = nim_data.n_partitions // rows

    # Calculate partition dimensions
    partition_height = nim_data.height // rows
    partition_width = nim_data.width // cols

    # Calculate leakage in pixels
    leakage = min(
        int(nim_data.leakage * partition_height),
        int(nim_data.leakage * partition_width),
    )

    # Draw partition boundaries
    for i in range(1, rows):
        ax.axhline(y=i * partition_height, color="blue", linestyle="--", alpha=0.5)
    for i in range(1, cols):
        ax.axvline(x=i * partition_width, color="blue", linestyle="--", alpha=0.5)

    # Draw leakage boundaries if leakage > 0
    if nim_data.leakage > 0:
        for i in range(1, rows):
            y_pos = i * partition_height
            ax.axhline(y=y_pos - leakage, color="red", linestyle=":", alpha=0.3)
            ax.axhline(y=y_pos + leakage, color="red", linestyle=":", alpha=0.3)
        for i in range(1, cols):
            x_pos = i * partition_width
            ax.axvline(x=x_pos - leakage, color="red", linestyle=":", alpha=0.3)
            ax.axvline(x=x_pos + leakage, color="red", linestyle=":", alpha=0.3)

    # Generate distinct colors for each partition
    colors = plt.cm.get_cmap("tab20")(np.linspace(0, 1, nim_data.n_partitions))

    # Plot sampled points from each partition
    for i, partition_data in enumerate(nim_data.get_partitions()):
        if partition_data:
            features = np.array([item[0] for item in partition_data]) * nim_data.feature_norm
            labels = np.array([item[1] for item in partition_data])

            # Plot points with different markers based on label
            for label_value in np.unique(labels):
                mask = labels == label_value
                mask = mask.flatten()
                ax.scatter(
                    features[mask, 0],
                    features[mask, 1],
                    color=colors[i],
                    alpha=0.7,
                    marker="o" if label_value == 1 else "x",
                    s=30,
                )

    # Create legend
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors[i],
            markersize=10,
            label=f"Partition {i + 1}",
        )
        for i in range(nim_data.n_partitions)
    ]
    # Add boundary lines to legend
    handles.append(Line2D([0], [0], color="blue", linestyle="--", label="Partition Boundaries"))
    if nim_data.leakage > 0:
        handles.append(Line2D([0], [0], color="red", linestyle=":", label="Leakage Boundaries"))

    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left")

    # ax.set_title(f"NIMData Partitions Visualization\n{nim_data.image_file}")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def _visualize_lidar_sampling(  # noqa: PLR0914, PLR0915
    nim_data: NIMDatasetHandler,
    *,
    path_index: int | list[int] | None = None,
    animate: bool = False,
    fps: int = 10,
    figsize: tuple[int, int] = (10, 8),
    save_path: str | None = None,
) -> FuncAnimation | Figure:
    """
    Visualize paths and simulated lidar hits for a `NIMData` instance.

    In static mode (animate=False) all paths are drawn on the image together with
    the lidar hit points stored in each partition.

    In animated mode (animate=True) the scanner(s) move along the chosen path(s)
    frame-by-frame, shooting beams and accumulating hits in real-time.  All agents
    are shown simultaneously.  The animation is returned and, if `save_path` is
    given, saved to disk.

    Args:
        nim_data: A NIMDatasetHandler instance that used path-based lidar sampling.
        path_index: Index or list of indices of the path(s) to focus on.  If None,
            all paths are shown / animated.
        animate: Whether to produce an animated scan instead of a static plot.
        fps: Frames per second used when saving the animation.
        figsize: Figure size.
        save_path: If given, the animation (or static figure) is saved here.

    Returns:
        A FuncAnimation when animate=True, otherwise a matplotlib Figure object.

    Raises:
        ValueError: If nim_data.paths is not set, or if path_index is invalid.

    """
    if not nim_data.paths:
        raise ValueError("NIMData.paths must be set to visualize lidar paths")

    # ------------------------------------------------------------------ helpers
    def _load_paths() -> list[list[tuple[int, int]]]:
        if isinstance(nim_data.paths, list):
            return nim_data.paths
        if isinstance(nim_data.paths, str):
            with Path(nim_data.paths).open("r", encoding="utf-8") as fh:
                return json.load(fh)  # type: ignore[no-any-return]
        raise ValueError("nim_data.paths must be a list of paths or a path to a JSON file")

    def _to_pixel(feature: Array) -> NDArray[np.float64]:
        """Convert a (possibly transformed) normalised feature back to pixel coords."""
        return np.asarray(feature, dtype=np.float64) * nim_data.feature_norm  # type: ignore[no-any-return]

    # ------------------------------------------------------------------ shared setup
    image = Image.open(nim_data.image_file).convert("L")
    img_array = np.array(image)
    img_float = img_array.astype(np.float64) / 255.0
    occ = image_to_occupancy(img_float, threshold=nim_data.occupancy_threshold)
    occ_bool = occ.astype(bool)
    occ_overlay = _make_red_occupancy_overlay(occ_bool, alpha=0.4)

    max_range = max(nim_data.height, nim_data.width) / 2 if nim_data.max_range is None else nim_data.max_range

    paths_list = _load_paths()
    n_paths = len(paths_list)
    colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, max(n_paths, 1)))

    partitions = nim_data.get_partitions()

    # ================================================================== ANIMATE
    if animate:
        # Normalise path_index to a list of indices
        if path_index is None:
            indices = list(range(n_paths))
        elif isinstance(path_index, int):
            indices = [path_index]
        else:
            indices = path_index

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(img_array, cmap="gray", origin="upper")
        ax.imshow(
            occ_overlay,
            origin="upper",
            interpolation="nearest",
        )
        label = f"Paths {indices}" if len(indices) > 1 else f"Path {indices[0]}"
        # ax.set_title(f"LiDAR Scan Animation – {label}\n{nim_data.image_file}")
        ax.axis("off")

        # Per-agent data and artists
        agent_scans: list[list[tuple[float, float, list[RayCastHit]]]] = []
        agent_dots: list[Line2D] = []
        agent_beam_segs: list[list[Line2D]] = []
        agent_hit_scatters: list[PathCollection] = []
        agent_hits_x: list[list[float]] = [[] for _ in indices]
        agent_hits_y: list[list[float]] = [[] for _ in indices]

        for pi in indices:
            color = colors[pi % len(colors)]
            raw_path = paths_list[pi]
            positions = densify_path(raw_path, nim_data.scan_spacing) if nim_data.scan_spacing else raw_path
            poses = compute_headings(positions)

            scans: list[tuple[float, float, list[RayCastHit]]] = []
            for ox, oy, heading in poses:
                beams = simulate_lidar_scan(
                    occ,
                    (ox, oy),
                    num_beams=nim_data.num_beams,
                    fov=nim_data.fov,
                    max_range=max_range,
                    heading=heading,
                )
                scans.append((ox, oy, beams))
            agent_scans.append(scans)

            # Static path overlay
            px = [p[0] for p in raw_path]
            py = [p[1] for p in raw_path]
            ax.plot(px, py, color=color, linewidth=1.5, alpha=0.4, zorder=2)

            # Per-agent mutable artists
            (dot,) = ax.plot([], [], "o", color=color, markersize=8, zorder=6)
            segs = [
                ax.plot([], [], "-", color=color, linewidth=0.8, alpha=0.6, zorder=4)[0]
                for _ in range(nim_data.num_beams)
            ]
            scatter = ax.scatter([], [], color=color, s=6, alpha=0.7, zorder=5)
            agent_dots.append(dot)
            agent_beam_segs.append(segs)
            agent_hit_scatters.append(scatter)

        all_artists: list[Line2D | PathCollection] = [
            *agent_dots,
            *agent_hit_scatters,
            *(seg for segs in agent_beam_segs for seg in segs),
        ]

        max_frames = max(len(s) for s in agent_scans)

        def _init() -> list[Line2D | PathCollection]:
            for dot in agent_dots:
                dot.set_data([], [])
            for segs in agent_beam_segs:
                for seg in segs:
                    seg.set_data([], [])
            for scatter in agent_hit_scatters:
                scatter.set_offsets(np.empty((0, 2)))
            return all_artists

        def _update(frame: int) -> list[Line2D | PathCollection]:
            for i, (dot, segs, scatter) in enumerate(zip(agent_dots, agent_beam_segs, agent_hit_scatters, strict=True)):
                agent_scans_i = agent_scans[i]
                if frame >= len(agent_scans_i):
                    continue
                ox, oy, beams = agent_scans_i[frame]
                dot.set_data([ox], [oy])

                agent_hits_x[i].clear()
                agent_hits_y[i].clear()
                for seg, hit in zip(segs, beams, strict=True):
                    seg.set_data([ox, hit.hit_point[0]], [oy, hit.hit_point[1]])
                    if hit.hit:
                        agent_hits_x[i].append(float(hit.hit_point[0]))
                        agent_hits_y[i].append(float(hit.hit_point[1]))

                if agent_hits_x[i]:
                    scatter.set_offsets(np.column_stack([agent_hits_x[i], agent_hits_y[i]]))
            return all_artists

        anim = FuncAnimation(
            fig,
            _update,
            frames=max_frames,
            init_func=_init,
            interval=1 / fps * 1000,
            blit=True,
        )

        if save_path:
            anim.save(save_path, fps=fps)

        plt.tight_layout()
        return anim

    # ================================================================== STATIC
    if path_index is None:
        path_indices = list(range(n_paths))
    elif isinstance(path_index, int):
        path_indices = [path_index]
    else:
        path_indices = path_index

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img_array, cmap="gray", origin="upper")
    ax.imshow(
        occ_overlay,
        origin="upper",
        interpolation="nearest",
    )
    ax.axis("off")

    handles: list[Line2D] = []
    for pi in path_indices:
        color = colors[pi % len(colors)]
        raw_path = paths_list[pi]

        # Draw path line
        px = [p[0] for p in raw_path]
        py = [p[1] for p in raw_path]
        ax.plot(px, py, color=color, linewidth=2, alpha=0.8, zorder=3)
        ax.plot(px[0], py[0], "o", color=color, markersize=8, zorder=4)  # start
        ax.plot(px[-1], py[-1], "s", color=color, markersize=8, zorder=4)  # end

        # Overlay lidar hit points stored in the corresponding partition
        if pi < len(partitions):
            partition = partitions[pi]
            if partition:
                features = np.array([_to_pixel(item[0]) for item in partition])
                labels = np.array([np.asarray(item[1]).flatten()[0] for item in partition])
                hit_mask = labels.astype(bool)
                if hit_mask.any():
                    ax.scatter(
                        features[hit_mask, 0],
                        features[hit_mask, 1],
                        color=color,
                        s=8,
                        alpha=0.7,
                        zorder=2,
                    )
                if (~hit_mask).any():
                    ax.scatter(
                        features[~hit_mask, 0],
                        features[~hit_mask, 1],
                        color=color,
                        s=8,
                        alpha=0.7,
                        marker="x",
                        zorder=2,
                    )

        handles.append(Line2D([0], [0], color=color, linewidth=2, label=f"Path {pi}"))

    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left")
    # ax.set_title(f"NIMData LiDAR Paths\n{nim_data.image_file}")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig
