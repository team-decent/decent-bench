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
from decent_bench.utils.checkpoint_manager import CheckpointManager
from decent_bench.utils.pytorch_utils import ArgmaxActivation, SimpleLinearModel
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks
from examples.nim.src.algorithms.lt_admm_ema import LT_ADMM_EMA

if __name__ == "__main__":
    iterations = 1000
    n_agents = 5
    n_neighbors = 4
    state_snapshot_period = 200
    samples_per_partition = 2000
    targets_per_partition = 2
    device = SupportedDevices.CPU
    use_dataloader = False
    compile_model = False
    opt_cls = torch.optim.Adam
    opt_kwargs = None
    sched_cls = None  # torch.optim.lr_scheduler.StepLR
    sched_kwargs = None  # {"step_size": 100, "gamma": 0.9}
    iop.set_seed(47)

    # EMA params
    ema_factors = [0.5, 0.7, 0.9]
    send_ema_x = False
    use_z_ema = False

    # All LT-ADMM tests
    penalty = [0.5, 1.0, 1.5]

    # ProxSkip
    step_size = 0.005

    # Final tests
    batch_sizes = [32, 64]
    local_steps = [10, 15]
    step_size = [0.01, 0.001]

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

    for heterogeneity in [True, False]:
        mnist_train = MNIST(root="data", train=True, download=True, transform=transform, target_transform=torch.tensor)
        mnist_test = MNIST(root="data", train=False, download=True, transform=transform, target_transform=torch.tensor)
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
            samples_per_partition=samples_per_partition // 2,
            heterogeneity=heterogeneity,
            targets_per_partition=targets_per_partition,
        )
        for alg in [
            "DGD",
            "LT-ADMM",
            "LT-ADMM-EMA",
            "LT-ADMM-EMA-OPT",
            "KGT2",
            "LED2",
            "ProxSkip",
            "ProxSkip2",
        ]:
            for batch_size in batch_sizes:
                resume_benchmark = False
                folder = "results/heterogeneous" if heterogeneity else "results/random"
                checkpoint_path = Path(f"{folder}/{alg}/bs_{batch_size}")
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

                if cm.is_completed():
                    print(f"Benchmark already completed for {checkpoint_path}. Skipping.")
                    continue

                costs = [
                    PyTorchCost(
                        dataset=p,
                        model=model_generator(),
                        loss_fn=nn.CrossEntropyLoss(),
                        final_activation=ArgmaxActivation(),
                        batch_size=batch_size,
                        max_batch_size=batch_size * 2,
                        device=device,
                        use_dataloader=use_dataloader,
                        dataloader_kwargs={} if device == SupportedDevices.GPU else {"pin_memory": True},
                        compile_model=compile_model,
                        compile_kwargs={"mode": "reduce-overhead"},
                    )
                    for p in train_dataset.get_partitions()
                ]
                print("Train set partitioned")
                agents = [Agent(i, cost, state_snapshot_period=state_snapshot_period) for i, cost in enumerate(costs)]
                graph = nx.random_regular_graph(d=n_neighbors, n=n_agents, seed=iop.get_seed())
                network = P2PNetwork(graph=graph, agents=agents)
                problem = benchmark.BenchmarkProblem(
                    network=network,
                    test_data=test_dataset.get_datapoints(),
                )
                x0 = pytorch_initialization(network, all_same=True)
                algorithms = []
                if alg == "DGD":
                    algorithms = []
                    for ss2 in [*step_size, 0.1, 0.5, 1.0]:
                        algorithms.extend(
                            dec_algorithms.DGD(
                                step_size=ss1,
                                aux_step_size=ss2,
                                iterations=iterations,
                                x0=x0,
                                name=f"DGD-ss1-{ss1}-ss2-{ss2}",
                            )
                            for ss1 in [*step_size, 0.1, 0.5]
                        )
                elif alg == "KGT":
                    algorithms = []
                    for ss1 in step_size:
                        for ss2 in step_size:
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
                elif alg == "KGT2":
                    algorithms = []
                    for ss1 in step_size:
                        for ss2 in [*step_size, 0.1, 0.5]:
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
                elif alg == "ProxSkip" or alg == "ProxSkip2":
                    algorithms = []
                    for ss1 in [*step_size, 0.1, 0.5]:
                        for ss2 in [*step_size, 0.1, 0.5]:
                            for comm_prob in [0.1, 0.5, 1.0]:
                                for chi in [1.0, 1.5, 2.0]:
                                    algorithms.append(
                                        dec_algorithms.ProxSkip(
                                            iterations=iterations,
                                            step_size=ss1,
                                            aux_step_size=ss2,
                                            comm_probability=comm_prob,
                                            chi=chi,
                                            x0=x0,
                                            name=f"ProxSkip-ss-{ss1}-aux_ss-{ss2}-comm_prob-{comm_prob}-chi-{chi}",
                                        )
                                    )
                elif alg == "LED":
                    algorithms = []
                    for ss1 in step_size:
                        for ss2 in step_size:
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
                elif alg == "LED2":
                    algorithms = []
                    for ss1 in step_size:
                        for ss2 in [*step_size, 0.1, 0.5]:
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
                    for p in penalty:
                        for ss1 in step_size:
                            for ss2 in step_size:
                                algorithms.extend(
                                    dec_algorithms.LT_ADMM(
                                        iterations=iterations,
                                        local_steps=ls,
                                        step_size=ss1,
                                        aux_step_size=ss2,
                                        penalty=p,
                                        x0=x0,
                                        name=f"LT-ADMM-ls-{ls}-ss1-{ss1}-ss2-{ss2}-p-{p}",
                                    )
                                    for ls in local_steps
                                )
                elif alg == "LT-ADMM-VR":
                    algorithms = []
                    for p in penalty:
                        for ss1 in step_size:
                            for ss2 in step_size:
                                algorithms.extend(
                                    dec_algorithms.LT_ADMM_VR(
                                        iterations=iterations,
                                        local_steps=ls,
                                        step_size=ss1,
                                        aux_step_size=ss2,
                                        penalty=p,
                                        x0=x0,
                                        name=f"LT-ADMM-VR-ls-{ls}-ss1-{ss1}-ss2-{ss2}-p-{p}",
                                    )
                                    for ls in local_steps
                                )
                elif alg == "LT-ADMM-EMA":
                    algorithms = []
                    for p in penalty:
                        for ss1 in step_size:
                            for ss2 in step_size:
                                for ema_factor in ema_factors:
                                    algorithms.extend(
                                        LT_ADMM_EMA(
                                            iterations=iterations,
                                            local_steps=ls,
                                            step_size=ss1,
                                            aux_step_size=ss2,
                                            penalty=p,
                                            ema_factor=ema_factor,
                                            send_ema_x=send_ema_x,
                                            use_z_ema=use_z_ema,
                                            x0=x0,
                                            name=f"LT-ADMM-EMA-ls-{ls}-ss1-{ss1}-ss2-{ss2}-p-{p}-ema-{ema_factor}",
                                        )
                                        for ls in local_steps
                                    )
                elif alg == "LT-ADMM-EMA-OPT":
                    algorithms = []
                    for p in penalty:
                        for ss1 in step_size:
                            for ss2 in step_size:
                                for ema_factor in ema_factors:
                                    algorithms.extend(
                                        LT_ADMM_EMA(
                                            iterations=iterations,
                                            local_steps=ls,
                                            step_size=ss1,
                                            aux_step_size=ss2,
                                            penalty=p,
                                            ema_factor=ema_factor,
                                            send_ema_x=send_ema_x,
                                            use_z_ema=use_z_ema,
                                            x0=x0,
                                            opt_cls=opt_cls,
                                            opt_kwargs=opt_kwargs,
                                            name=f"LT-ADMM-EMA-OPT-ls-{ls}-ss1-{ss1}-ss2-{ss2}-p-{p}-ema-{ema_factor}",
                                        )
                                        for ls in local_steps
                                    )

                algorithms = sorted(algorithms, key=lambda alg: alg.name)

                if alg == "ProxSkip":
                    # Take first half
                    algorithms = algorithms[: len(algorithms) // 2]
                elif alg == "ProxSkip2":
                    # Take second half
                    algorithms = algorithms[len(algorithms) // 2 :]

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
                        n_trials=2 if alg == "ProxSkip" else 1,
                        show_speed=True,
                        show_trial=True,
                        checkpoint_manager=cm,
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
                del result
                del metric_result
                # Garbage collection to free up memory before the next benchmark
                gc.collect()

    print("All benchmarks completed.")
