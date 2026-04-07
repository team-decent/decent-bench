import ctypes
import gc
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from src.dataset import NIMDatasetHandler
from torch import nn

import decent_bench.algorithms.decentralized as dec_algorithms
import decent_bench.utils.interoperability as iop
from decent_bench import benchmark
from decent_bench.agents import Agent
from decent_bench.algorithms.utils import pytorch_initialization
from decent_bench.costs import PyTorchCost
from decent_bench.metrics import metric_library as ml
from decent_bench.metrics import runtime_library
from decent_bench.networks import P2PNetwork
from decent_bench.schemes import GaussianNoise, TopK, UniformActivationRate, UniformDropRate
from decent_bench.utils.checkpoint_manager import CheckpointManager
from decent_bench.utils.pytorch_utils import SimpleLinearModel
from decent_bench.utils.types import SupportedDevices
from examples.nim.src.algorithms.lt_admm_ema import LT_ADMM_EMA
from examples.nim.src.algorithms.lt_admm_torch_optimizer import LT_ADMM_TORCH


def _trim_process_memory() -> None:
    """Best-effort return of freed heap pages back to the OS on glibc systems."""
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except OSError:
        return


if __name__ == "__main__":
    iterations = 10_000
    state_snapshot_period = 100
    samples_per_partition = 2000
    test_samples = 10_000
    leakage = 0.0
    label_balance = 2.0
    image_file = "data/kth_floorplan.png"
    batch_size = 32
    local_steps = [5, 10]
    device = SupportedDevices.CPU
    opt_cls = torch.optim.Adam

    table_metrics = [
        ml.ConsensusError([min, np.average, max]),
        ml.GradientCalls([np.average, sum]),
        ml.SentMessages([np.average, sum]),
        ml.MSE([min, np.average, max]),
        ml.Loss([min, np.average, max]),
    ]

    plot_metrics = [
        [ml.ConsensusError([], x_log=False, y_log=True)],
        [ml.MSE([], x_log=False, y_log=True)],
        [ml.Loss([], x_log=False, y_log=False)],
    ]

    def model_generator() -> torch.nn.Module:
        """Generate a simple linear model for the NIM dataset."""
        return SimpleLinearModel(
            input_size=2,
            hidden_sizes=[32, 16],
            output_size=1,
        )

    class FinalActivation(nn.Module):  # noqa: D101
        def __init__(self, threshold: float = 0.5):
            super().__init__()
            self.sigmoid = nn.Sigmoid()
            self.threshold = threshold

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
            return (self.sigmoid(x) > self.threshold).long()

    for n_agents, n_neighbors in [(5, 4), (5, 2)]:
        for drops, activity, compression, noise in [
            (True, True, True, True),
            (True, None, None, None),
            (None, True, None, None),
            (None, None, True, None),
            (None, None, None, True),
            (None, None, None, None),
        ]:
            for alg in [
                "DGD",
                "KGT",
                "LED",
                "ProxSkip",
                "LT-ADMM",
                "LT-ADMM-TORCH",
                "LT-ADMM-VR",
                "LT-ADMM-EMA",
                "LT-ADMM-EMA-TORCH",
            ]:
                iop.set_seed(47)
                train_dataset = NIMDatasetHandler(
                    image_file=image_file,
                    n_partitions=n_agents,
                    samples_per_partition=samples_per_partition,
                    transform=torch.tensor,  # type: ignore[arg-type]
                    label_transform=lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(0),  # type: ignore[arg-type]
                    label_balance=label_balance,
                    leakage=leakage,
                )

                resume_benchmark = False
                folder = "results/nim"
                checkpoint_path = Path(
                    f"{folder}/{n_agents}_{n_neighbors}/test_{drops}_{activity}_{compression}_{noise}/{alg}"
                )
                if checkpoint_path.exists():
                    resume_benchmark = True

                cm = CheckpointManager(
                    checkpoint_dir=checkpoint_path,
                    checkpoint_step=iterations // 2,
                    keep_n_checkpoints=1,
                    benchmark_metadata={
                        "dataset": "NIM",
                        "n_agents": n_agents,
                        "n_neighbors": n_neighbors,
                        "drops": drops,
                        "activity": activity,
                        "compression": compression,
                        "noise": noise,
                    },
                )

                is_benchmark_completed = cm.is_benchmark_completed()
                if is_benchmark_completed and cm.are_metrics_computed():
                    print(f"Benchmark already completed for {checkpoint_path}. Skipping.")
                    continue

                costs = [
                    PyTorchCost(
                        dataset=p,
                        model=model_generator(),
                        loss_fn=nn.BCEWithLogitsLoss(),
                        final_activation=FinalActivation(threshold=0.5),
                        batch_size=batch_size,
                        max_batch_size=batch_size * 4,
                        device=device,
                    )
                    for p in train_dataset.get_partitions()
                ]
                agents = [
                    Agent(
                        i,
                        cost,
                        state_snapshot_period=state_snapshot_period,
                        activation=UniformActivationRate(0.8) if activity else None,
                    )
                    for i, cost in enumerate(costs)
                ]
                graph = nx.random_regular_graph(d=n_neighbors, n=n_agents, seed=iop.get_seed())
                network = P2PNetwork(
                    graph=graph,
                    agents=agents,
                    message_noise=GaussianNoise(0.0, 0.01) if noise else None,
                    message_compression=TopK(0.1) if compression else None,
                    message_drop=UniformDropRate(0.2) if drops else None,
                )
                problem = benchmark.BenchmarkProblem(
                    network=network,
                    test_data=train_dataset.get_test_set(label_balance=1.0, num_samples=test_samples),
                )
                x0 = pytorch_initialization(network, all_same=True)
                if alg == "DGD":
                    algorithms = [
                        dec_algorithms.DGD(
                            step_size=0.1,
                            aux_step_size=1.0,
                            iterations=iterations,
                            x0=x0,
                        ),
                    ]
                elif alg == "KGT":
                    algorithms = [
                        dec_algorithms.KGT(
                            iterations=iterations,
                            local_steps=ls,
                            step_size=0.01,
                            aux_step_size=0.5,
                            x0=x0,
                            name=f"KGT (ls={ls})",
                        )
                        for ls in local_steps
                    ]
                elif alg == "ProxSkip":
                    algorithms = [
                        dec_algorithms.ProxSkip(
                            iterations=iterations,
                            step_size=0.1,
                            aux_step_size=0.1,
                            comm_probability=1.0 / ls,
                            chi=1.0,
                            x0=x0,
                            name=f"ProxSkip (p={1.0 / ls:.2f})",
                        )
                        for ls in local_steps
                    ]
                elif alg == "LED":
                    algorithms = [
                        dec_algorithms.LED(
                            iterations=iterations,
                            local_steps=ls,
                            step_size=0.01,
                            aux_step_size=0.01,
                            x0=x0,
                            name=f"LED (ls={ls})",
                        )
                        for ls in local_steps
                    ]
                elif alg == "LT-ADMM":
                    algorithms = [
                        dec_algorithms.LT_ADMM(
                            iterations=iterations,
                            local_steps=ls,
                            step_size=0.01,
                            aux_step_size=0.01,
                            penalty=1.0,
                            mask_z=False,
                            x0=x0,
                            name=f"LT-ADMM (ls={ls})",
                        )
                        for ls in local_steps
                    ]
                elif alg == "LT-ADMM-TORCH":
                    algorithms = [
                        LT_ADMM_TORCH(
                            iterations=iterations,
                            local_steps=ls,
                            step_size=0.001,
                            aux_step_size=0.01,
                            penalty=1.0,
                            mask_z=False,
                            x0=x0,
                            opt_cls=opt_cls,
                            name=f"LT-ADMM-TORCH (ls={ls})",
                        )
                        for ls in local_steps
                    ]
                elif alg == "LT-ADMM-VR":
                    algorithms = [
                        dec_algorithms.LT_ADMM_VR(
                            iterations=iterations,
                            local_steps=ls,
                            step_size=0.01,
                            aux_step_size=0.01,
                            penalty=1.0,
                            mask_z=False,
                            x0=x0,
                            name=f"LT-ADMM-VR (ls={ls})",
                        )
                        for ls in local_steps
                    ]
                elif alg == "LT-ADMM-EMA":
                    algorithms = [
                        LT_ADMM_EMA(
                            iterations=iterations,
                            local_steps=ls,
                            step_size=0.005,
                            aux_step_size=0.01,
                            penalty=1.0,
                            ema_factor=0.8,
                            send_ema_x=False,
                            use_z_ema=False,
                            mask_z=False,
                            x0=x0,
                            name=f"LT-ADMM-EMA (ls={ls})",
                        )
                        for ls in local_steps
                    ]
                elif alg == "LT-ADMM-EMA-TORCH":
                    algorithms = [
                        LT_ADMM_EMA(
                            iterations=iterations,
                            local_steps=ls,
                            step_size=0.0005,
                            aux_step_size=0.01,
                            penalty=1.0,
                            ema_factor=0.8,
                            send_ema_x=False,
                            use_z_ema=False,
                            mask_z=False,
                            x0=x0,
                            opt_cls=opt_cls,
                            name=f"LT-ADMM-EMA-TORCH (ls={ls})",
                        )
                        for ls in local_steps
                    ]

                algorithms = sorted(algorithms, key=lambda alg: alg.name)

                if not is_benchmark_completed:
                    if resume_benchmark:
                        result = benchmark.resume_benchmark(
                            checkpoint_manager=cm,
                            create_backup=False,
                            show_speed=True,
                            show_trial=True,
                            runtime_metrics=[runtime_library.RuntimeLoss(250)],
                        )
                    else:
                        result = benchmark.benchmark(
                            algorithms=algorithms,
                            benchmark_problem=problem,
                            n_trials=3 if alg == "ProxSkip" or any((drops, activity, noise)) else 1,
                            show_speed=True,
                            show_trial=True,
                            checkpoint_manager=cm,
                            runtime_metrics=[runtime_library.RuntimeLoss(250)],
                        )
                else:
                    result = None

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

                metric_result.agent_metrics = None
                del result, metric_result, costs, agents, network, problem, algorithms
                # Garbage collection to free up memory before the next benchmark
                gc.collect()
                _trim_process_memory()

    print("All benchmarks completed.")
