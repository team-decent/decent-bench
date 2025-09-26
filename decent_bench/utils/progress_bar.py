from dataclasses import dataclass
from multiprocessing.managers import SyncManager
from queue import Queue
from threading import Thread

from rich.progress import BarColumn, Progress, TaskID, TaskProgressColumn, TextColumn, TimeRemainingColumn

from decent_bench.distributed_algorithms import DstAlgorithm


@dataclass(eq=False)
class _ProgressRecord:
    progress_bar_id: TaskID
    increment: int


class ProgressBarController:
    """
    Controller of progress bars showing how far each algorithm has progressed and the estimated time remaining.

    Args:
        manager: used to create a progress increment queue that can be shared across processes
        algorithms: algorithms that will be run, each gets its own bar
        n_trials: number of trials the algorithms will run

    """

    def __init__(self, manager: SyncManager, algorithms: list[DstAlgorithm], n_trials: int):
        self._progress_increment_queue: Queue[_ProgressRecord] = manager.Queue()
        orchestrator = Progress(
            TextColumn("{task.description}"),
            BarColumn(finished_style="bold green", pulse_style="none"),
            TaskProgressColumn(),
            TimeRemainingColumn(elapsed_when_finished=True),
            speed_estimate_period=300,
        )
        self._progress_bar_ids = {alg: orchestrator.add_task(alg.name, total=n_trials) for alg in algorithms}
        orchestrator.start()
        listener_thread = Thread(target=self._progress_listener, args=(orchestrator, self._progress_increment_queue))
        listener_thread.start()

    def start_progress_bar(self, algorithm: DstAlgorithm) -> None:
        """
        Start the clock of *algorithm*'s progress bar without incrementing it.

        Internally, this is done through sending an increment of 0 to the progress listener. The progress listener
        recognizes that the algorithm's execution just started and resets its clock, which started when the progress bar
        was first rendered.
        """
        progress_bar_id = self._progress_bar_ids[algorithm]
        self._progress_increment_queue.put(_ProgressRecord(progress_bar_id, 0))

    def advance_progress_bar(self, algorithm: DstAlgorithm) -> None:
        """Advance *algorithm*'s progress bar by one trial."""
        progress_bar_id = self._progress_bar_ids[algorithm]
        self._progress_increment_queue.put(_ProgressRecord(progress_bar_id, 1))

    @staticmethod
    def _progress_listener(orchestrator: Progress, queue: Queue[_ProgressRecord]) -> None:
        started_progress_bar_ids = set()
        while not orchestrator.finished:
            progress_record = queue.get()
            if progress_record.progress_bar_id not in started_progress_bar_ids:
                orchestrator.reset(progress_record.progress_bar_id)
                started_progress_bar_ids.add(progress_record.progress_bar_id)
            orchestrator.advance(progress_record.progress_bar_id, progress_record.increment)
        orchestrator.stop()
