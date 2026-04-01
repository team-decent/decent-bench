from pathlib import Path

import networkx as nx
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST

import decent_bench.utils.interoperability as iop
from decent_bench import benchmark
from decent_bench.agents import Agent
from decent_bench.algorithms import decentralized
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
    folder = "results/heterogeneous_2"
    iterations = 1000
    n_trials = 3
    n_agents = 5
    n_neighbors = 4
    state_snapshot_period = 50
    samples_per_partition = 1000
    heterogeneity = False
    targets_per_partition = 2
    batch_sizes = [32, 64, 128]
    local_steps = [1, 5, 10]
    step_sizes = [0.01, 0.001]
    comm_probs = [0.2, 0.5, 0.9]
    device = SupportedDevices.CPU
    use_dataloader = False
    compile_model = False
    opt_cls = torch.optim.Adam
    opt_kwargs = None
    sched_cls = None  # torch.optim.lr_scheduler.StepLR
    sched_kwargs = {"step_size": 100, "gamma": 0.9}
    iop.set_seed(47)

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
        samples_per_partition=samples_per_partition,
        heterogeneity=heterogeneity,
        targets_per_partition=targets_per_partition,
    )

    def model_generator() -> torch.nn.Module:
        """Generate a simple linear model for the MNIST dataset."""
        return SimpleLinearModel(
            input_size=train_dataset.n_features,
            hidden_sizes=[32, 16],
            output_size=train_dataset.n_targets,
        )

    for batch_size in batch_sizes:
        for ss in step_sizes:
            checkpoint_path = Path(
                f"{folder}/mnist_bs_{batch_size}_ss_{ss}_hg_{heterogeneity}_tp_{targets_per_partition}"
            )
            if checkpoint_path.exists():
                print(f"Checkpoint already exists at {checkpoint_path}. Skipping benchmark.")
                continue

            cm = CheckpointManager(
                checkpoint_dir=checkpoint_path,
                checkpoint_step=iterations // 3,
                benchmark_metadata={
                    "dataset": "MNIST",
                    "model": "SimpleLinearModel",
                    "n_agents": n_agents,
                    "n_neighbors": n_neighbors,
                    "heterogeneity": heterogeneity,
                    "targets_per_partition": targets_per_partition,
                    "batch_size": batch_size,
                    "step_size": ss,
                },
            )

            costs = [
                PyTorchCost(
                    dataset=p,
                    model=model_generator(),
                    loss_fn=nn.CrossEntropyLoss(),
                    final_activation=ArgmaxActivation(),
                    batch_size=batch_size,
                    max_batch_size=batch_size,
                    device=device,
                    use_dataloader=use_dataloader,
                    dataloader_kwargs={} if device == SupportedDevices.GPU else {"pin_memory": True},
                    compile_model=compile_model,
                    compile_kwargs={"mode": "reduce-overhead"},
                )
                for p in train_dataset.get_partitions()
            ]
            agents = [Agent(i, cost, state_snapshot_period=state_snapshot_period) for i, cost in enumerate(costs)]
            graph = nx.random_regular_graph(d=n_neighbors, n=n_agents, seed=iop.get_seed())
            network = P2PNetwork(graph=graph, agents=agents)
            problem = benchmark.BenchmarkProblem(
                network=network,
                test_data=list(test_dataset.get_datapoints()),  # hacky fix to make it pickleable
            )
            x0 = pytorch_initialization(network)
            algorithms = []

            algorithms.extend(
                [
                    decentralized.DGD(
                        iterations=iterations,
                        x0=x0,
                        step_size=ss,
                    )
                ],
            )
            for comm_prob in comm_probs:
                algorithms.extend(
                    [
                        decentralized.ProxSkip(
                            iterations=iterations,
                            x0=x0,
                            step_size=ss,
                            aux_step_size=ss,
                            comm_probability=comm_prob,
                            name=f"ProxSkip-{comm_prob}CP",
                        ),
                    ],
                )
            for ls in local_steps:
                algorithms.extend(
                    [
                        decentralized.KGT(
                            iterations=iterations,
                            local_steps=ls,
                            step_size=ss,
                            aux_step_size=ss,
                            x0=x0,
                            name=f"KGT-{ls}LS",
                        ),
                        decentralized.LED(
                            iterations=iterations,
                            x0=x0,
                            step_size=ss,
                            aux_step_size=ss,
                            local_steps=ls,
                            name=f"LED-{ls}LS",
                        ),
                        decentralized.LT_ADMM(
                            iterations=iterations,
                            x0=x0,
                            step_size=ss,
                            local_steps=ls,
                            name=f"LT-ADMM-{ls}LS",
                        ),
                        decentralized.LT_ADMM_VR(
                            iterations=iterations,
                            x0=x0,
                            step_size=ss,
                            local_steps=ls,
                            name=f"LT-ADMM-VR-{ls}LS",
                        ),
                        LT_ADMM_EMA(
                            iterations=iterations,
                            x0=x0,
                            step_size=ss,
                            local_steps=ls,
                            send_ema_x=False,
                            name=f"LT-ADMM-EMA-Optim-F-{ls}LS",
                        ),
                        LT_ADMM_EMA(
                            iterations=iterations,
                            x0=x0,
                            step_size=ss,
                            local_steps=ls,
                            opt_cls=opt_cls,
                            opt_kwargs=opt_kwargs,
                            sched_cls=sched_cls,
                            sched_kwargs=sched_kwargs,
                            send_ema_x=False,
                            name=f"LT-ADMM-EMA-Torch-F-{ls}LS",
                        ),
                        LT_ADMM_EMA(
                            iterations=iterations,
                            x0=x0,
                            step_size=ss,
                            local_steps=ls,
                            send_ema_x=True,
                            name=f"LT-ADMM-EMA-Optim-T-{ls}LS",
                        ),
                        LT_ADMM_EMA(
                            iterations=iterations,
                            x0=x0,
                            step_size=ss,
                            local_steps=ls,
                            opt_cls=opt_cls,
                            opt_kwargs=opt_kwargs,
                            sched_cls=sched_cls,
                            sched_kwargs=sched_kwargs,
                            send_ema_x=True,
                            name=f"LT-ADMM-EMA-Torch-T-{ls}LS",
                        ),
                    ],
                )

            algorithms = sorted(algorithms, key=lambda alg: alg.name)

            result = benchmark.benchmark(
                algorithms=algorithms,
                benchmark_problem=problem,
                n_trials=n_trials,
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
