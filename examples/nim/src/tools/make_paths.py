"""
Interactive utility to draw agent paths on an image and save them to a JSON file.

Usage:
    python make_paths.py /path/to/image.png --out /path/to/paths.json

Controls (when the matplotlib window is active):
 - Left click: add a point to the current agent path
 - Right click: finish the current agent path (start a new path)
 - 'u' key: undo last point in current path
 - 'i' key: save current paths to output file
 - 'g' key: toggle grid lines for better alignment
 - 'q' or close window: quit without saving (unless already saved)

The output JSON is a list of agents; each agent is a list of [x,y] coordinates (floats, pixel coords).
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.backend_bases import Event, KeyEvent, MouseEvent
from numpy.typing import NDArray
from PIL import Image

TITLE_TEXT = (
    "Left click to add point, right click to finish an agent path."
    "\n'i' to save, 'u' undo, 'g' to toggle grid, 'q' to quit"
)


def run(image_path: Path, out_path: Path) -> None:  # noqa: D103, PLR0915
    img = Image.open(image_path).convert("RGB")
    img_arr = np.array(img)

    fig, ax = plt.subplots()
    ax.imshow(img_arr)
    ax.set_title(TITLE_TEXT)

    paths: list[list[tuple[float, float]]] = []
    current_path: list[tuple[float, float]] = []
    showing_grid: bool = False

    colormap = cm.get_cmap("tab10")

    def redraw() -> None:
        ax.clear()
        ax.imshow(img_arr)
        # restore title (so it doesn't disappear after first click)
        ax.set_title(TITLE_TEXT)
        ax.grid(showing_grid)
        # Increase grid lines
        ax.set_xticks(np.arange(0, img_arr.shape[1], 50))
        ax.set_yticks(np.arange(0, img_arr.shape[0], 50))

        # draw completed paths with distinct colors
        for idx, p in enumerate(paths):
            if len(p) >= 1:
                xs = [pt[0] for pt in p]
                ys = [pt[1] for pt in p]
                color = colormap(idx % colormap.N)
                ax.plot(xs, ys, marker="o", linestyle="-", color=color, linewidth=2)
        # draw current path
        if current_path:
            xs = [pt[0] for pt in current_path]
            ys = [pt[1] for pt in current_path]
            ax.plot(xs, ys, marker="x", linestyle="--", color="red")
        fig.canvas.draw_idle()

    def on_click(event: Event) -> None:
        if not isinstance(event, MouseEvent):
            print("Non-mouse event received, ignoring.")  # noqa: T201
            return

        # ignore clicks outside axes
        if event.inaxes != ax:
            return

        if event.button == 1:
            # left click: add point
            if event.xdata is not None and event.ydata is not None:
                current_path.append((int(event.xdata), int(event.ydata)))
                redraw()
            else:
                print("Click was outside image bounds, ignoring.")  # noqa: T201
        elif event.button == 3:
            # right click: finish current path if it has points
            if current_path:
                paths.append(_connect_loop(current_path.copy(), img_arr))
                current_path.clear()
                redraw()
            else:
                print(  # noqa: T201
                    "No points in current path to finish, ignoring right click."
                )

    def on_key(event: Event) -> None:
        nonlocal showing_grid

        if not isinstance(event, KeyEvent):
            print("Non-key event received, ignoring.")  # noqa: T201
            return

        key = event.key
        if key == "u":
            if current_path:
                current_path.pop()
                redraw()
            elif paths:
                current_path.extend(paths.pop())
                current_path.pop()  # remove last point from popped path to undo the finish action
                redraw()
            else:
                print("No points or paths to undo.")  # noqa: T201
        elif key == "i":
            # Save: append current path if non-empty
            if current_path:
                paths.append(_connect_loop(current_path.copy(), img_arr))
                current_path.clear()
            out_path.parent.mkdir(parents=True, exist_ok=True)

            showing_grid = False  # turn off grid for clean output
            redraw()

            with out_path.open("w", encoding="utf-8") as f:
                json.dump(paths, f, indent=2)
            print(f"Saved {len(paths)} paths to {out_path}")  # noqa: T201
        elif key == "g":
            # Show grid lines for better alignment
            showing_grid = not showing_grid
            redraw()
        elif key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    print("Interactive drawing started.")  # noqa: T201
    print(TITLE_TEXT)  # noqa: T201
    plt.show()


def _connect_loop(
    path: list[tuple[float, float]],
    image: NDArray[np.float64],
) -> list[tuple[float, float]]:
    """
    Check if the path forms a loop.

    If the path has 4 or more points, check if the first and last point are within a certain distance of each other,
    and if so, connect the last and first to make a loop.
    """
    if len(path) < 4:
        return path

    h, w = image.shape[:2]
    first_pt = path[0]
    last_pt = path[-1]
    dist = np.hypot(last_pt[0] - first_pt[0], last_pt[1] - first_pt[1])
    if dist < max(h, w) * 0.03:  # threshold in pixels to consider it a loop
        path[-1] = first_pt  # connect last point to first

    return path


def main() -> None:  # noqa: D103
    parser = argparse.ArgumentParser(description="Draw agent paths on an image and save to JSON.")
    parser.add_argument("image", type=str, help="Path to image file")
    parser.add_argument("--out", type=str, default="paths.json", help="Output JSON file")
    args = parser.parse_args()
    run(Path(args.image), Path(args.out if ".json" in args.out else args.out + ".json"))


if __name__ == "__main__":
    main()
