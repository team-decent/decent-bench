import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import NDArray
from PIL import Image

from .dataset import NIMDatasetHandler, image_to_occupancy


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


def visualize_nim_dataset(  # noqa: PLR0914
    nim_data: NIMDatasetHandler,
    num_samples_per_partition: int = 100,
    figsize: tuple[int, int] = (12, 10),
    save_path: str | None = None,
) -> Figure:
    """
    Visualize the spatial partitions and sampled points from a NIMData dataset.

    Args:
        nim_data: A NIMData instance
        num_samples_per_partition: Number of points to sample from each partition for visualization.
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
        int(nim_data.overlap * partition_height),
        int(nim_data.overlap * partition_width),
    )

    # Draw partition boundaries
    for i in range(1, rows):
        ax.axhline(y=i * partition_height, color="blue", linestyle="--", alpha=0.5)
    for i in range(1, cols):
        ax.axvline(x=i * partition_width, color="blue", linestyle="--", alpha=0.5)

    # Draw leakage boundaries if leakage > 0
    if nim_data.overlap > 0:
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
        p_data = random.sample(partition_data, min(num_samples_per_partition, len(partition_data)))
        features = np.array([item[0] for item in p_data]) * nim_data._feature_norm  # noqa: SLF001
        labels = np.array([item[1] for item in p_data])

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
    if nim_data.overlap > 0:
        handles.append(Line2D([0], [0], color="red", linestyle=":", label="Leakage Boundaries"))

    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left")

    ax.set_title(f"NIMData Partitions Visualization\n{nim_data.image_file}")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig
