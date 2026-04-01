import numpy as np

from decent_bench import benchmark
from decent_bench.metrics import metric_library as ml
from decent_bench.utils.checkpoint_manager import CheckpointManager

if __name__ == "__main__":
    iterations = 1000
    batch_size = 32
    ss = 0.01
    heterogeneity = False
    targets_per_partition = 2
    backup = input("Create backup before resuming? (y/n): ").lower() == "y"

    table_metrics = [
        ml.ConsensusError([min, np.average, max]),
        ml.GradientCalls([np.average, sum]),
        ml.SentMessages([np.average, sum]),
        ml.Accuracy([min, np.average, max], fmt=".2%"),
        ml.Precision([min, np.average, max], fmt=".2%"),
        ml.Recall([min, np.average, max], fmt=".2%"),
        ml.Loss([min, np.average, max]),
    ]

    plot_metrics = [
        ml.ConsensusError([], x_log=False, y_log=True),
        ml.Accuracy([], x_log=False, y_log=False),
        ml.Precision([], x_log=False, y_log=False),
        ml.Recall([], x_log=False, y_log=False),
        ml.Loss([], x_log=False, y_log=False),
    ]

    cm = CheckpointManager(
        checkpoint_dir=f"results/mnist_bs_{batch_size}_ss_{ss}_hg_{heterogeneity}_tp_{targets_per_partition}",
        checkpoint_step=iterations // 5,
    )

    result = benchmark.resume_benchmark(
        checkpoint_manager=cm,
        show_speed=True,
        show_trial=True,
        create_backup=backup,
    )

    metric_result = benchmark.compute_metrics(
        benchmark_result=result,
        checkpoint_manager=cm,
        table_metrics=table_metrics,
        plot_metrics=plot_metrics,
    )

    benchmark.display_metrics(
        metrics_result=metric_result,
        checkpoint_manager=cm,
        show_plots=False,
    )
