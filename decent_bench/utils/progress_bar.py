from collections.abc import Iterable
from dataclasses import dataclass
from math import ceil
from multiprocessing.managers import SyncManager
from queue import Queue
from threading import Thread
from typing import TYPE_CHECKING, Any

from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Column, Table
from rich.text import Text

from decent_bench.distributed_algorithms import Algorithm

if TYPE_CHECKING:
    from rich.progress import Task


@dataclass(eq=False)
class _ProgressRecord:
    """Record of progress bar update to be sent to the progress listener."""

    progress_bar_id: TaskID
    increment: int
    trial: int | None


class TrialColumn(ProgressColumn):
    """Safe column that shows 'Trial [X/total]' without KeyError if field missing."""

    def __init__(self, n_trials: int, style: str = "", finished_style: str = "") -> None:
        super().__init__()
        self.n_trials = n_trials
        self.style = style
        self.finished_style = finished_style

    def render(self, task: "Task") -> Text:  # noqa: D102
        trial = self.n_trials if task.finished else task.fields.get("fields", task.fields).get("trial", "?")
        return Text(f"{trial}/{self.n_trials}", style=self.finished_style if task.finished else self.style)


class SpeedColumn(ProgressColumn):
    """Column that shows speed in iterations/second."""

    def __init__(self, progress_step: int | None) -> None:
        super().__init__()
        self.progress_step = progress_step

    def render(self, task: "Task") -> Text:  # noqa: D102
        if task.speed is None and task.finished_speed is None:
            return Text("--.-- it/s", style="progress.percentage", justify="right")

        speed = task.finished_speed or task.speed
        if speed is not None and self.progress_step is not None:
            # Normalize speed to iterations/second depending on progress step
            speed *= self.progress_step

        text = TaskProgressColumn.render_speed(speed)
        text.justify = "right"
        return text


class ProgressWithHeader(Progress):
    """Custom Progress display with column headers."""

    def __init__(  # type: ignore[no-untyped-def]
        self,
        *columns: str | ProgressColumn,
        headers: Iterable[Text] | None = None,
        **kwargs,  # noqa: ANN003
    ) -> None:
        self.headers = headers
        super().__init__(*columns, **kwargs)

    def make_tasks_table(self, tasks: Iterable["Task"]) -> Table:
        """Override to add header row to the table."""
        if not tasks or not self.headers:
            return super().make_tasks_table(tasks)

        # Mimic super() implementation but render headers of columns
        table_columns = [
            (Column(no_wrap=True) if isinstance(col, str) else col.get_table_column().copy()) for col in self.columns
        ]
        for col, header in zip(table_columns, self.headers, strict=False):
            col.header = header

        table = Table(
            *table_columns,
            padding=(0, 1),
            expand=self.expand,
            show_header=True,
            box=None,
            collapse_padding=True,
            show_footer=False,
            show_edge=False,
            pad_edge=False,
        )

        # Add each task as a row
        for task in tasks:
            if task.visible:
                table.add_row(
                    *(
                        (column.format(task=task) if isinstance(column, str) else column(task))
                        for column in self.columns
                    )
                )

        return table


@dataclass
class ProgressBarHandle:
    """
    A picklable handle for worker processes to update :class:`ProgressBarController`.

    This class contains only the picklable parts needed by worker processes,
    separating them from the unpicklable Thread components in ProgressBarController.
    """

    _progress_increment_queue: Queue[Any]
    _progress_bar_ids: dict[Any, Any]
    _progress_step: int | None

    def start_progress_bar(self, algorithm: Algorithm, trial: int) -> None:
        """
        Start the clock of *algorithm*'s progress bar without incrementing it.

        Internally, this is done through sending an increment of 0 to the progress listener. The progress listener
        recognizes that the algorithm's execution just started and resets its clock, which started when the progress bar
        was first rendered.
        """
        progress_bar_id = self._progress_bar_ids[algorithm]
        self._progress_increment_queue.put(_ProgressRecord(progress_bar_id, 0, trial + 1))

    def advance_progress_bar(self, algorithm: Algorithm, iteration: int) -> None:
        """Advance *algorithm*'s progress bar by an amount (units)."""
        if self._progress_step is None:
            if (iteration + 1) < algorithm.iterations:
                return
        elif (iteration + 1) % self._progress_step != 0 and (iteration + 1) < algorithm.iterations:
            return

        progress_bar_id = self._progress_bar_ids[algorithm]
        self._progress_increment_queue.put(_ProgressRecord(progress_bar_id, 1, None))


class ProgressBarController:
    """
    Controller of progress bars showing how far each algorithm has progressed and the estimated time remaining.

    Args:
        manager: A multiprocessing :class:`~multiprocessing.managers.SyncManager` instance used to create a shared queue
            for coordinating progress updates across multiple processes. This enables thread-safe communication between
            worker processes and the progress bar listener thread.
        algorithms: algorithms that will be run, each gets its own bar
        n_trials: number of trials the algorithms will run
        progress_step: if provided, the progress bar will step every `progress_step`.
            When provided, each algorithm's task total becomes `n_trials * ceil(algorithm.iterations / progress_step)`.
            If `None`, the progress bar uses 1 unit per trial.

    Note:
        If `progress_step` is too small performance may degrade due to the
        overhead of updating the progress bar too often.

    """

    def __init__(  # noqa: PLR0917
        self,
        manager: SyncManager,
        algorithms: list[Algorithm],
        n_trials: int,
        progress_step: int | None,
        show_speed: bool = False,
        show_trial: bool = False,
    ):
        self._progress_increment_queue: Queue[_ProgressRecord | None] = manager.Queue()
        self.progress_step = progress_step
        p_cols = [
            (TextColumn("{task.description}"), Text("Algorithm", style="bold")),
            (BarColumn(finished_style="bold green", pulse_style="none"), Text("Progress Bar", style="bold")),
            (TaskProgressColumn(), Text("", style="bold")),  # Skip % Completed header as it's part of progress bar
            *([(SpeedColumn(progress_step), Text("Speed", style="bold"))] if show_speed else []),
            (TimeRemainingColumn(elapsed_when_finished=True), Text("Time", style="bold")),
            *(
                [
                    (
                        TrialColumn(n_trials=n_trials, style="progress.remaining", finished_style="progress.elapsed"),
                        Text("Active Trials", style="bold"),
                    )
                ]
                if show_trial
                else []
            ),
        ]

        cols, headers = zip(*p_cols, strict=True)
        orchestrator = ProgressWithHeader(*cols, headers=headers)
        if progress_step is None:
            self._progress_bar_ids = {alg: orchestrator.add_task(alg.name, total=n_trials) for alg in algorithms}
        else:
            steps_per_trial = {alg: max(1, ceil(alg.iterations / progress_step)) for alg in algorithms}
            self._progress_bar_ids = {
                alg: orchestrator.add_task(alg.name, total=n_trials * steps_per_trial[alg]) for alg in algorithms
            }

        orchestrator.start()
        self.listener_thread = Thread(
            target=self._progress_listener, args=(orchestrator, self._progress_increment_queue)
        )
        self.listener_thread.start()
        self._handle = ProgressBarHandle(
            _progress_increment_queue=self._progress_increment_queue,
            _progress_bar_ids=self._progress_bar_ids,
            _progress_step=self.progress_step,
        )

    def get_handle(self) -> ProgressBarHandle:
        """
        Get a picklable handle for worker processes.

        Returns a handle containing only the queue and metadata needed by worker
        processes, without the unpicklable Thread component.
        """
        return self._handle

    def stop(self) -> None:
        """Stop the progress bar and wait for the listener thread to finish."""
        # Signal the listener thread to stop if it is still running
        # because an exception occurred in one algorithm's execution.
        self._progress_increment_queue.put(None)
        if hasattr(self, "listener_thread"):
            self.listener_thread.join(timeout=2.0)

    @staticmethod
    def _progress_listener(orchestrator: Progress, queue: Queue[_ProgressRecord | None]) -> None:
        started_progress_bar_ids = set()
        while not orchestrator.finished:
            progress_record = queue.get()
            if progress_record is None:
                break

            if progress_record.progress_bar_id not in started_progress_bar_ids:
                orchestrator.reset(progress_record.progress_bar_id)
                started_progress_bar_ids.add(progress_record.progress_bar_id)
            if progress_record.trial is not None:
                orchestrator.update(progress_record.progress_bar_id, fields={"trial": str(progress_record.trial)})
            orchestrator.advance(progress_record.progress_bar_id, progress_record.increment)
        orchestrator.stop()
