import contextlib
import math
import queue
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from decent_bench.metrics._plots import _get_marker_style_color
from decent_bench.utils.logger import LOGGER

if TYPE_CHECKING:
    from multiprocessing.context import DefaultContext, Process, SpawnContext, SpawnProcess


class RuntimeMetricPlotter:
    """
    Centralized plotter for runtime metrics that runs in its own process.

    This class handles all matplotlib plotting operations in a separate process,
    receiving data from other processes via a queue. This avoids X server conflicts
    when using multiprocessing.

    Note:
        This class is not intended to be instantiated by users. It is automatically
        created and managed by the benchmark infrastructure.

    """

    def __init__(self, queue: "queue.Queue[Any]", context: "SpawnContext | DefaultContext") -> None:
        """
        Initialize the runtime metric plotter.

        Args:
            queue: Multiprocessing queue for receiving plot data from other processes.
                   Format: (metric_id, algorithm_name, trial, iteration, value)
            context: Multiprocessing context to use for creating the process.

        """
        self._queue = queue
        self._context = context
        self._process: Process | SpawnProcess | None = None
        self._figures: dict[str, Figure] = {}  # metric_id -> figure
        self._axes: dict[str, Axes] = {}  # metric_id -> axes
        self._lines: dict[tuple[str, str, int], Line2D] = {}  # (metric_id, alg_name, trial) -> line
        self._data: dict[tuple[str, str, int], tuple[list[int], list[float]]] = {}
        self._modified_figures: set[str] = set()  # Track which figures need redrawing
        self._algorithm_trial_colors: dict[tuple[str, int], int] = {}  # (alg_name, trial) -> color index
        self._should_save_plots: dict[str, Path] = {}  # metric_id -> Path to save plot (if any)

    def start(self) -> None:
        """Start the plotter in a separate process using the provided context."""
        # Create a Process using the context
        self._process = self._context.Process(target=self.run, daemon=True)
        self._process.start()

    def run(self) -> None:  # noqa: PLR0912
        """Process loop, continuously process queue updates."""
        # Set matplotlib to use a backend that works in a separate process
        plt.ion()

        try:  # noqa: PLR1702
            while True:
                processed = 0
                try:
                    # Block waiting for first message
                    data = self._queue.get(timeout=0.05)

                    # Check for stop sentinel
                    if data == "STOP":
                        break

                    if data is not None:
                        self._process_message(data)
                        processed += 1

                    # Process all remaining messages without blocking (batch processing)
                    while True:
                        try:
                            data = self._queue.get_nowait()
                            if data == "STOP":
                                return
                            if data is not None:
                                self._process_message(data)
                                processed += 1

                            if processed >= 200:
                                # If we processed a large batch, redraw to update the plots
                                self._draw_modified_figures()
                                plt.pause(0.001)
                                processed = 0
                        except queue.Empty:
                            # Queue is empty
                            break

                    # Redraw modified figures and update GUI once after processing batch
                    if processed > 0:
                        self._draw_modified_figures()
                        plt.pause(0.001)

                except KeyboardInterrupt:
                    # Allow graceful shutdown on Ctrl+C
                    self.shutdown(dont_save=True)
                    break
                except queue.Empty:
                    # Timeout or other error - small pause to allow GUI updates
                    plt.pause(0.01)
                    continue
                except Exception as e:
                    LOGGER.debug(f"Error in RuntimeMetricPlotter: {e}")
                    plt.pause(0.01)
                    continue
        finally:
            self._close_all()

    def create_figure(self, metric_id: str, description: str, save_path: Path | None) -> None:
        """
        Create a figure for a metric.

        Args:
            metric_id: Unique identifier for the metric
            description: Human-readable description for the y-axis label
            save_path: Path to save the plot when updated (if None, no saving is performed)

        """
        if metric_id in self._figures:
            return

        if save_path is not None:
            self._should_save_plots[metric_id] = save_path

        fig, ax = plt.subplots()
        ax.set_xlabel("Iteration")
        ax.set_ylabel(description)
        ax.set_title(f"{description} - All Trials")
        ax.grid(visible=True, alpha=0.3)

        self._figures[metric_id] = fig
        self._axes[metric_id] = ax

        plt.show(block=False)

    def update(self, metric_id: str, algorithm_name: str, trial: int, iteration: int, value: float) -> None:
        """
        Update a plot with new data.

        Args:
            metric_id: Unique identifier for the metric
            algorithm_name: Name of the algorithm
            trial: Trial number
            iteration: Current iteration
            value: Metric value

        """
        if metric_id not in self._figures:
            return

        if math.isnan(value) or math.isinf(value):
            LOGGER.warning(
                f"Received invalid value for metric '{metric_id}' from algorithm "
                f"'{algorithm_name}' trial {trial} at iteration {iteration}: {value}"
            )
            return

        key = (metric_id, algorithm_name, trial)
        # Get or create data lists
        if key not in self._data:
            self._data[key] = ([], [])
        x_data, y_data = self._data[key]
        x_data.append(iteration)
        y_data.append(value)

        # Create or update line
        if key not in self._lines:
            ax = self._axes[metric_id]
            sub_key = (algorithm_name, trial)
            if sub_key not in self._algorithm_trial_colors:
                self._algorithm_trial_colors[sub_key] = len(self._algorithm_trial_colors)
            marker, linestyle, color = _get_marker_style_color(self._algorithm_trial_colors[sub_key])
            (line,) = ax.plot(
                x_data,
                y_data,
                label=f"{algorithm_name} - Trial {trial}",
                linewidth=1.5,
                color=color,
                marker=marker,
                linestyle=linestyle,
                markevery=0.1,
                alpha=0.7,
            )
            self._lines[key] = line
            # Sort legend entries alphabetically
            handles, labels = ax.get_legend_handles_labels()
            sorted_pairs = sorted(zip(labels, handles, strict=True))
            sorted_labels, sorted_handles = zip(*sorted_pairs, strict=True) if sorted_pairs else ([], [])
            ax.legend(sorted_handles, sorted_labels, loc="best", fontsize=8)
        else:
            line = self._lines[key]
            line.set_data(x_data, y_data)

        # Mark for redrawing (actual draw happens after batch processing)
        self._modified_figures.add(metric_id)
        ax = self._axes[metric_id]
        ax.relim()
        ax.autoscale_view()

    def shutdown(self, dont_save: bool = False) -> None:
        """
        Signal the plotter process to stop and wait for it to finish.

        Args:
            dont_save: If True, do not save plots to files on shutdown even if save paths were provided.

        Note:
            This method can be called multiple times safely. If the process is already stopped, it will do nothing.
            If the process is still running, it will send a stop signal and wait for it to finish.
            If the process does not stop within a reasonable time, it will be forcefully terminated.

        """
        if self._queue is not None:
            with contextlib.suppress(Exception):
                self._queue.put("STOP")
        if self._process is not None:
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.terminate()
        self._close_all(dont_save=dont_save)

    def _process_message(self, data: Any) -> None:  # noqa: ANN401
        """Process a single message from the queue."""
        try:
            if not isinstance(data, (tuple, list)):
                LOGGER.debug(f"Received invalid message in RuntimeMetricPlotter: {data}")
                return

            # Handle different message types
            if len(data) == 4 and data[0] == "init":
                # Initialization message: ("init", metric_id, description)
                try:
                    _, metric_id, description, save_path = data
                except (TypeError, ValueError):
                    LOGGER.debug("Ignoring malformed init message: %r", data)
                    return
                self.create_figure(metric_id, description, save_path)
            elif len(data) == 5:
                # Data update message: (metric_id, algorithm_name, trial, iteration, value)
                try:
                    metric_id, algorithm_name, trial, iteration, value = data
                except (TypeError, ValueError):
                    LOGGER.debug("Ignoring malformed update message: %r", data)
                    return
                self.update(metric_id, algorithm_name, trial, iteration, value)
            else:
                LOGGER.debug(f"Received message with unrecognized format in RuntimeMetricPlotter: {data}")
        except Exception as e:
            LOGGER.debug(f"Error processing message in RuntimeMetricPlotter: {e}")

    def _draw_modified_figures(self) -> None:
        """Redraw all figures that were modified during batch processing."""
        for metric_id in self._modified_figures:
            if metric_id in self._figures:
                self._figures[metric_id].canvas.draw()
                self._figures[metric_id].canvas.flush_events()
        self._modified_figures.clear()

    def _draw_all_figures(self) -> None:
        """Redraw all figures."""
        for fig in self._figures.values():
            fig.canvas.draw()
            fig.canvas.flush_events()
        plt.pause(0.01)

    def _close_all(self, dont_save: bool = False) -> None:
        """
        Close all figure windows.

        Args:
            dont_save: If True, do not save plots to files even if save paths were provided.
            This can be used for a quick shutdown without saving when plots are not needed.

        """
        if len(self._should_save_plots) > 0 and not dont_save:
            self._draw_all_figures()  # Ensure all figures are up to date before saving

            for metric_id in self._should_save_plots:
                if metric_id in self._figures:
                    save_path = self._should_save_plots[metric_id]
                    save_path.mkdir(parents=True, exist_ok=True)
                    self._figures[metric_id].savefig(save_path / f"{metric_id}.png")
                    LOGGER.info(f"Saved plot for metric '{metric_id}' to {save_path / f'{metric_id}.png'}")

        for fig in self._figures.values():
            plt.close(fig)
        self._figures.clear()
        self._axes.clear()
        self._lines.clear()
        self._data.clear()
        self._should_save_plots.clear()
