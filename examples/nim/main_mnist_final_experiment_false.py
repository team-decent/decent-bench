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
from decent_bench.schemes import GaussianNoise, Quantization, UniformActivationRate, UniformDropRate
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
    state_snapshot_period = 250
    samples_per_partition = 500
    device = SupportedDevices.CPU
    use_dataloader = False
    compile_model = False
    opt_cls = torch.optim.Adam
    iop.set_seed(47)

    # Params
    ss1s = [0.01, 0.001]
    ss2s = [0.5, 0.1, 0.01, 0.001]
    admm_ss2s = [0.01, 0.001]
    dgd_step_sizes = [1.0, 0.5, 0.1, 0.01, 0.001]
    proxskip_step_size = [1.0, 0.5, 0.1, 0.01]
    local_steps = [5, 10, 15, 20]
    ema_factors = [0.9, 0.8, 0.7, 0.6]
    penalties_chi = 1.0
    batch_size = 32

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
                (None, None, None, None),
                (True, None, None, None),
                (None, True, None, None),
                (None, None, True, None),
                (None, None, None, True),
                (True, True, True, True),
            ]:
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
                    samples_per_partition=samples_per_partition,
                    heterogeneity=heterogeneity,
                    targets_per_partition=targets_per_partition,
                )
                for alg in [
                    "DGD",
                    "KGT",
                    "LED",
                    "ProxSkip",
                    "LT-ADMM",
                    "LT-ADMM-EMA",
                    "LT-ADMM-EMA-OPT",
                    "LT-ADMM-TORCH",
                    "LT-ADMM-VR",
                ]:
                    resume_benchmark = False
                    folder = "results/heterogeneous" if heterogeneity else "results/random"
                    checkpoint_path = Path(
                        f"{folder}/{n_agents}_{n_neighbors}_{drops}_{activity}_{compression}_{noise}/{alg}"
                    )
                    if checkpoint_path.exists():
                        resume_benchmark = True

                    cm = CheckpointManager(
                        checkpoint_dir=checkpoint_path,
                        checkpoint_step=iterations // 3,
                        keep_n_checkpoints=2,
                        benchmark_metadata={
                            "dataset": "MNIST",
                            "model": "SimpleLinearModel",
                            "n_agents": n_agents,
                            "n_neighbors": n_neighbors,
                            "heterogeneity": heterogeneity,
                            "targets_per_partition": targets_per_partition,
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
                            use_dataloader=use_dataloader,
                            dataloader_kwargs={} if device == SupportedDevices.GPU else {"pin_memory": True},
                            compile_model=compile_model,
                            compile_kwargs={"mode": "reduce-overhead"},
                        )
                        for p in train_dataset.get_partitions()
                    ]
                    print("Train set partitioned")
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
                        message_compression=Quantization(6) if compression else None,
                        message_drop=UniformDropRate(0.1) if drops else None,
                    )
                    problem = benchmark.BenchmarkProblem(
                        network=network,
                        test_data=test_dataset.get_datapoints(),
                    )
                    x0 = pytorch_initialization(network, all_same=True)
                    algorithms = []
                    if alg == "DGD":
                        algorithms = []
                        for ss2 in dgd_step_sizes:
                            algorithms.extend(
                                dec_algorithms.DGD(
                                    step_size=ss1,
                                    aux_step_size=ss2,
                                    iterations=iterations,
                                    x0=x0,
                                    name=f"DGD-ss1-{ss1}-ss2-{ss2}",
                                )
                                for ss1 in dgd_step_sizes
                            )
                    elif alg == "KGT":
                        algorithms = []
                        for ss1 in ss1s:
                            for ss2 in ss2s:
                                algorithms.extend(
                                    dec_algorithms.KGT(
                                        iterations=iterations,
                                        local_steps=ls,
                                        step_size=ss1,
                                        aux_step_size=ss2,
                                        x0=x0,
                                        name=f"KGT-ls-{ls}-ss1-{ss1}-ss2-{ss2}",
                                    )
                                    for ls in local_steps
                                )
                    elif alg == "ProxSkip":
                        algorithms = []
                        for ss1 in proxskip_step_size:
                            for ss2 in proxskip_step_size:
                                for ls in local_steps:
                                    algorithms.append(
                                        dec_algorithms.ProxSkip(
                                            iterations=iterations,
                                            step_size=ss1,
                                            aux_step_size=ss2,
                                            comm_probability=1.0 / ls,
                                            chi=penalties_chi,
                                            x0=x0,
                                            name=f"ProxSkip-ss-{ss1}-aux_ss-{ss2}-comm_prob-{1.0 / ls}",
                                        )
                                    )
                    elif alg == "LED":
                        algorithms = []
                        for ss1 in ss1s:
                            for ss2 in ss2s:
                                algorithms.extend(
                                    dec_algorithms.LED(
                                        iterations=iterations,
                                        local_steps=ls,
                                        step_size=ss1,
                                        aux_step_size=ss2,
                                        x0=x0,
                                        name=f"LED-ls-{ls}-ss1-{ss1}-ss2-{ss2}",
                                    )
                                    for ls in local_steps
                                )
                    elif alg == "LT-ADMM":
                        algorithms = []
                        for ss1 in ss1s:
                            for ss2 in admm_ss2s:
                                algorithms.extend(
                                    dec_algorithms.LT_ADMM(
                                        iterations=iterations,
                                        local_steps=ls,
                                        step_size=ss1,
                                        aux_step_size=ss2,
                                        penalty=penalties_chi,
                                        x0=x0,
                                        name=f"LT-ADMM-ls-{ls}-ss1-{ss1}-ss2-{ss2}",
                                    )
                                    for ls in local_steps
                                )
                    elif alg == "LT-ADMM-VR":
                        algorithms = []
                        for ss1 in ss1s:
                            for ss2 in admm_ss2s:
                                algorithms.extend(
                                    dec_algorithms.LT_ADMM_VR(
                                        iterations=iterations,
                                        local_steps=ls,
                                        step_size=ss1,
                                        aux_step_size=ss2,
                                        penalty=penalties_chi,
                                        x0=x0,
                                        name=f"LT-ADMM-VR-ls-{ls}-ss1-{ss1}-ss2-{ss2}",
                                    )
                                    for ls in local_steps
                                )
                    elif alg == "LT-ADMM-EMA":
                        algorithms = []
                        for ss1 in ss1s:
                            for ss2 in admm_ss2s:
                                for ema_factor in ema_factors:
                                    algorithms.extend(
                                        LT_ADMM_EMA(
                                            iterations=iterations,
                                            local_steps=ls,
                                            step_size=ss1,
                                            aux_step_size=ss2,
                                            penalty=penalties_chi,
                                            ema_factor=ema_factor,
                                            send_ema_x=False,
                                            use_z_ema=False,
                                            x0=x0,
                                            name=f"LT-ADMM-EMA-ls-{ls}-ss1-{ss1}-ss2-{ss2}-ema-{ema_factor}",
                                        )
                                        for ls in local_steps
                                    )
                    elif alg == "LT-ADMM-EMA-OPT":
                        algorithms = []
                        for ss1 in ss1s:
                            for ss2 in admm_ss2s:
                                for ema_factor in ema_factors:
                                    algorithms.extend(
                                        LT_ADMM_EMA(
                                            iterations=iterations,
                                            local_steps=ls,
                                            step_size=ss1,
                                            aux_step_size=ss2,
                                            penalty=penalties_chi,
                                            ema_factor=ema_factor,
                                            send_ema_x=False,
                                            use_z_ema=False,
                                            x0=x0,
                                            opt_cls=opt_cls,
                                            name=f"LT-ADMM-EMA-OPT-ls-{ls}-ss1-{ss1}-ss2-{ss2}-ema-{ema_factor}",
                                        )
                                        for ls in local_steps
                                    )
                    elif alg == "LT-ADMM-TORCH":
                        algorithms = []
                        for ss1 in ss1s:
                            for ss2 in admm_ss2s:
                                algorithms.extend(
                                    LT_ADMM_TORCH(
                                        iterations=iterations,
                                        local_steps=ls,
                                        step_size=ss1,
                                        aux_step_size=ss2,
                                        penalty=penalties_chi,
                                        x0=x0,
                                        opt_cls=opt_cls,
                                        name=f"LT-ADMM-TORCH-ls-{ls}-ss1-{ss1}-ss2-{ss2}",
                                    )
                                    for ls in local_steps
                                )

                    algorithms = sorted(algorithms, key=lambda alg: alg.name)

                    if not is_benchmark_completed:
                        if resume_benchmark:
                            print(f"Resuming benchmark for {checkpoint_path}...")
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
                                # More trials for ProxSkip to account for its stochasticity
                                n_trials=1,
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
                    metric_result.agent_metrics = None
                    metric_result.plot_results = None
                    metric_result.table_results = None
                    metric_result.plot_metrics = None
                    metric_result.table_metrics = None
                    del result, metric_result, costs, agents, network, problem, algorithms
                    # Garbage collection to free up memory before the next benchmark
                    gc.collect()
                    _trim_process_memory()

    print("All benchmarks completed.")
