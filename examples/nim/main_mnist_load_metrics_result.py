import numpy as np

from decent_bench import benchmark
from decent_bench.metrics import metric_library as ml
from decent_bench.utils.checkpoint_manager import CheckpointManager

if __name__ == "__main__":

    cm = CheckpointManager(
        checkpoint_dir=r"/home/ubuntu/github/decent-bench/examples/nim/results/heterogeneous_2/mnist_bs_32_ss_0.01_hg_True_tp_2",
    )

    benchmark.display_metrics(
        checkpoint_manager=cm,
        show_plots=True,
    )
