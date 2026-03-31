import logging
import pathlib
import shutil

import numpy as np

from decent_bench import benchmark, benchmark_problem
from decent_bench.costs import LinearRegressionCost, LogisticRegressionCost, PyTorchCost
from decent_bench.distributed_algorithms import (
    ADMM,
    ATC,
    ATG,
    DGD,
    DLM,
    ED,
    EXTRA,
    NIDS,
    ATCTracking,
    AugDGM,
    SimpleGT,
    WangElia,
)
from decent_bench.metrics import ComputationalCost, runtime_collection
from decent_bench.metrics import metric_utils as utils
from decent_bench.metrics.metric_collection import (
    DEFAULT_PLOT_METRICS,
    DEFAULT_TABLE_METRICS,
    MSE,
    Accuracy,
    GradientCalls,
    Precision,
    Recall,
    Regret,
)
from decent_bench.utils.checkpoint_manager import CheckpointManager
from src.algorithms.lt_admm_ema import LT_ADMM_EMA

if __name__ == "__main__":
    iterations = 2000
    trials = 2
    checkpoint_path = pathlib.Path(
        f"benchmark_results/checkpoints-{iterations}-{trials}"
    )
    cost_cls = PyTorchCost
    etm = [
        Accuracy([min, np.average, max], fmt=".2%"),
        MSE([min, np.average, max]),
        Precision([min, np.average, max], fmt=".2%"),
        Recall([min, np.average, max], fmt=".2%"),
    ]
    epm = [
        Accuracy([], x_log=False, y_log=False, common_iterations=True),
        MSE([], x_log=False, y_log=True, common_iterations=True),
        Precision([], x_log=False, y_log=False, common_iterations=True),
        Recall([], x_log=False, y_log=False, common_iterations=True),
    ]
    checkpoint_manager = CheckpointManager(checkpoint_path, checkpoint_step=50)

    problem = benchmark_problem.create_classification_problem(
        cost_cls,
        n_agents=4,
        agent_state_snapshot_period=5,
    )
    x0 = (
        problem.costs[0]._get_model_parameters()  # noqa: SLF001
        if isinstance(problem.costs[0], PyTorchCost)
        else None
    )
    print(f"Running with x0: {type(x0)}")  # noqa: T201

    def lr_sched(i: int) -> float:
        return 0.01 * (0.99 ** (i // 100))

    result = None
    result = benchmark.benchmark(
        algorithms=[
            # DGD(iterations=iterations, step_size=0.01),
            # ATC(iterations=iterations, step_size=0.01),
            # na.GT_SARAH(iterations=iterations, local_steps=5, step_size=0.01),
            LT_ADMM_EMA(
                iterations=iterations,
                local_steps=5,
                step_size=lr_sched,
                penalty=1.0,
                alpha=0.5,
                ema_factor=0.9,
                set_x_to_ema=True,
                use_z_ema=True,
                use_torch_optim=True,
                x0=x0,
            ),
        ],
        benchmark_problem=problem,
        n_trials=trials,
        progress_step=10,
        show_speed=True,
        show_trial=True,
        max_processes=1 if cost_cls is PyTorchCost else None,
        runtime_metrics=[
            # runtime_collection.RuntimeLoss(
            #     20,
            #     checkpoint_manager.get_results_path(),
            # ),
            # runtime_collection.RuntimeRegret(
            #     20,
            #     checkpoint_manager.get_results_path(),
            # ),
            # runtime_collection.RuntimeGradientNorm(
            #     20,
            #     checkpoint_manager.get_results_path(),
            # ),
            # runtime_collection.RuntimeConsensusError(
            #     20,
            #     checkpoint_manager.get_results_path(),
            # ),
        ],
        # checkpoint_manager=checkpoint_manager,
    )

    # met = None
    # met = benchmark.compute_metrics(
    #     result,
    #     checkpoint_manager=checkpoint_manager,
    #     table_metrics=DEFAULT_TABLE_METRICS + etm,
    #     # plot_metrics=DEFAULT_PLOT_METRICS + epm,
    # )

    # benchmark.display_metrics(
    #     met,
    #     checkpoint_manager=checkpoint_manager,
    #     computational_cost=ComputationalCost(gradient=2.0),
    #     individual_plots=False,
    #     # table_metrics=[],
    #     # plot_metrics=[
    #     #     [
    #     #         Accuracy([], x_log=False, y_log=False, common_iterations=True),
    #     #         MSE([], x_log=False, y_log=True, common_iterations=True),
    #     #     ],
    #     #     [
    #     #         Precision([], x_log=False, y_log=False, common_iterations=True),
    #     #     ],
    #     #     [
    #     #         Recall([], x_log=False, y_log=False, common_iterations=True),
    #     #     ],
    #     # ],
    #     compare_iterations_and_computational_cost=False,
    #     plot_grid=False,
    #     plot_format="png",
    #     save_path=checkpoint_manager.get_results_path(),
    # )
