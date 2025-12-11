from dataclasses import dataclass
from math import ceil
from multiprocessing.managers import SyncManager
from queue import Queue
from threading import Thread
from typing import TYPE_CHECKING

from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text

from decent_bench.distributed_algorithms import Algorithm

if TYPE_CHECKING:
    from rich.progress import Task


@dataclass(eq=False)
class _ProgressRecord:
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
        if task.finished:
            trial = self.n_trials
        else:
            trial = (
                task.fields["fields"].get("trial", "?") if "fields" in task.fields else task.fields.get("trial", "?")
            )
        return Text(f"Trial [{trial}/{self.n_trials}]", style=self.finished_style if task.finished else self.style)


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


class ProgressBarController:
    """
    Controller of progress bars showing how far each algorithm has progressed and the estimated time remaining.

    Args:
        manager: used to create a progress increment queue that can be shared across processes
        algorithms: algorithms that will be run, each gets its own bar
        n_trials: number of trials the algorithms will run
        progress_step: if provided, the progress bar will step every `progress_step`.
            When provided, each algorithm's task total becomes `n_trials * ceil(algorithm.iterations / progress_step)`.
            If `None`, the progress bar uses 1 unit per trial.

    Note:
        If `progress_step` is too small performance may degrade due to the
        overhead of updating the progress bar too often.

    """

    def __init__(
        self,
        manager: SyncManager,
        algorithms: list[Algorithm],
        n_trials: int,
        progress_step: int | None,
        show_speed: bool = True,
    ):
        self._progress_increment_queue: Queue[_ProgressRecord] = manager.Queue()
        self.progress_step = progress_step
        p_cols = [
            TextColumn("{task.description}"),
            BarColumn(finished_style="bold green", pulse_style="none"),
            TaskProgressColumn(),
            *([SpeedColumn(progress_step)] if show_speed else []),
            TimeRemainingColumn(elapsed_when_finished=True),
            TrialColumn(n_trials=n_trials, style="progress.remaining", finished_style="progress.elapsed"),
        ]

        orchestrator = Progress(*p_cols)
        if progress_step is None:
            self._progress_bar_ids = {alg: orchestrator.add_task(alg.name, total=n_trials) for alg in algorithms}
        else:
            steps_per_trial = {alg: max(1, ceil(alg.iterations / progress_step)) for alg in algorithms}
            self._progress_bar_ids = {
                alg: orchestrator.add_task(alg.name, total=n_trials * steps_per_trial[alg]) for alg in algorithms
            }

        orchestrator.start()
        listener_thread = Thread(target=self._progress_listener, args=(orchestrator, self._progress_increment_queue))
        listener_thread.start()

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
        if self.progress_step is None:
            if (iteration + 1) < algorithm.iterations:
                return
        elif (iteration + 1) % self.progress_step != 0 and (iteration + 1) < algorithm.iterations:
            return

        progress_bar_id = self._progress_bar_ids[algorithm]
        self._progress_increment_queue.put(_ProgressRecord(progress_bar_id, 1, None))

    @staticmethod
    def _progress_listener(orchestrator: Progress, queue: Queue[_ProgressRecord]) -> None:
        started_progress_bar_ids = set()
        while not orchestrator.finished:
            progress_record = queue.get()
            if progress_record.progress_bar_id not in started_progress_bar_ids:
                orchestrator.reset(progress_record.progress_bar_id)
                started_progress_bar_ids.add(progress_record.progress_bar_id)
            if progress_record.trial is not None:
                orchestrator.update(progress_record.progress_bar_id, fields={"trial": str(progress_record.trial)})
            orchestrator.advance(progress_record.progress_bar_id, progress_record.increment)
        orchestrator.stop()
