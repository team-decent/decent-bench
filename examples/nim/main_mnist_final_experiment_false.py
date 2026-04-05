import ctypes
import gc
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST

import decent_bench.algorithms.decentralized as dec_algorithms
import decent_bench.utils.interoperability as iop
from decent_bench import benchmark
from decent_bench.agents import Agent
from decent_bench.algorithms.utils import pytorch_initialization
from decent_bench.costs import PyTorchCost
from decent_bench.datasets import PyTorchDatasetHandler
from decent_bench.metrics import metric_library as ml
from decent_bench.networks import P2PNetwork
from decent_bench.schemes import GaussianNoise, TopK, UniformActivationRate, UniformDropRate
from decent_bench.utils.checkpoint_manager import CheckpointManager
from decent_bench.utils.pytorch_utils import ArgmaxActivation, SimpleLinearModel
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks
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
    iterations = 1000
    state_snapshot_period = 50
    samples_per_partition = 1000
    batch_size = 32
    local_steps = [5, 10, 15]
    device = SupportedDevices.CPU
    opt_cls = torch.optim.Adam

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
        [ml.ConsensusError([], x_log=False, y_log=True)],
        [ml.Accuracy([], x_log=False, y_log=False)],
        [ml.Precision([], x_log=False, y_log=False)],
        [ml.Recall([], x_log=False, y_log=False)],
        [ml.Loss([], x_log=False, y_log=False)],
    ]

    torch_device = iop.device_to_framework_device(device, SupportedFrameworks.PYTORCH)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])

    def model_generator() -> torch.nn.Module:
        """Generate a simple linear model for the MNIST dataset."""
        return SimpleLinearModel(
            input_size=28 * 28,
            hidden_sizes=[32, 16],
            output_size=10,
        )

    for heterogeneity in [False]:
        for n_agents, n_neighbors in [(5, 4), (5, 2), (10, 9), (10, 4), (10, 2)]:
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
                    mnist_train = MNIST(
                        root="data",
                        train=True,
                        download=True,
                        transform=transform,
                        target_transform=torch.tensor,
                    )
                    mnist_test = MNIST(
                        root="data",
                        train=False,
                        download=True,
                        transform=transform,
                        target_transform=torch.tensor,
                    )
                    targets_per_partition = 2 if n_agents == 5 else 1
                    train_dataset = PyTorchDatasetHandler(
                        torch_dataset=mnist_train,
                        n_features=28 * 28,
                        n_targets=10,
                        n_partitions=n_agents,
                        samples_per_partition=samples_per_partition,
                        heterogeneity=heterogeneity,
                        targets_per_partition=targets_per_partition,
                    )
                    test_dataset = PyTorchDatasetHandler(
                        torch_dataset=mnist_test,
                        n_features=28 * 28,
                        n_targets=10,
                        n_partitions=n_agents,
                        heterogeneity=heterogeneity,
                        targets_per_partition=targets_per_partition,
                    )

                    resume_benchmark = False
                    folder = "results/heterogeneous" if heterogeneity else "results/random"
                    checkpoint_path = Path(
                        f"{folder}/{n_agents}_{n_neighbors}/test_{drops}_{activity}_{compression}_{noise}/{alg}"
                    )
                    if checkpoint_path.exists():
                        resume_benchmark = True

                    cm = CheckpointManager(
                        checkpoint_dir=checkpoint_path,
                        checkpoint_step=None,
                        keep_n_checkpoints=1,
                        benchmark_metadata={
                            "dataset": "MNIST",
                            "model": "SimpleLinearModel",
                            "n_agents": n_agents,
                            "n_neighbors": n_neighbors,
                            "heterogeneity": heterogeneity,
                            "targets_per_partition": targets_per_partition,
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
                            loss_fn=nn.CrossEntropyLoss(),
                            final_activation=ArgmaxActivation(),
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
                        message_compression=TopK(int(agents[0].cost.shape[0] * 0.7)) if compression else None,
                        message_drop=UniformDropRate(0.2) if drops else None,
                    )
                    problem = benchmark.BenchmarkProblem(
                        network=network,
                        test_data=test_dataset.get_datapoints(),
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
                                step_size=ss1,
                                aux_step_size=0.01,
                                penalty=1.0,
                                ema_factor=0.8,
                                send_ema_x=False,
                                use_z_ema=False,
                                x0=x0,
                                name=f"LT-ADMM-EMA (ls={ls}, ss={ss1})",
                            )
                            for ls in local_steps
                            for ss1 in [0.01, 0.005]
                        ]
                    elif alg == "LT-ADMM-EMA-TORCH":
                        algorithms = [
                            LT_ADMM_EMA(
                                iterations=iterations,
                                local_steps=ls,
                                step_size=ss1,
                                aux_step_size=0.01,
                                penalty=1.0,
                                ema_factor=0.8,
                                send_ema_x=False,
                                use_z_ema=False,
                                x0=x0,
                                opt_cls=opt_cls,
                                name=f"LT-ADMM-EMA-TORCH (ls={ls}, ss={ss1})",
                            )
                            for ls in local_steps
                            for ss1 in [0.001, 0.0005]
                        ]

                    algorithms = sorted(algorithms, key=lambda alg: alg.name)

                    if not is_benchmark_completed:
                        if resume_benchmark:
                            result = benchmark.resume_benchmark(
                                checkpoint_manager=cm,
                                create_backup=False,
                                show_speed=True,
                                show_trial=True,
                            )
                        else:
                            result = benchmark.benchmark(
                                algorithms=algorithms,
                                benchmark_problem=problem,
                                n_trials=3,
                                show_speed=True,
                                show_trial=True,
                                checkpoint_manager=cm,
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

                    del result, metric_result, costs, agents, network, problem, algorithms
                    # Garbage collection to free up memory before the next benchmark
                    gc.collect()
                    _trim_process_memory()

    print("All benchmarks completed.")
