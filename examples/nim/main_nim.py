import gc
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from PIL import Image
from src.algorithms.lt_admm_ema import LT_ADMM_EMA
from src.dataset import NIMDatasetHandler
from src.nim_helpers import NimModel
from torch import nn

import decent_bench.algorithms.decentralized as dec_algorithms
import decent_bench.utils.interoperability as iop
from decent_bench import benchmark
from decent_bench.agents import Agent
from decent_bench.algorithms.utils import pytorch_initialization
from decent_bench.costs import PyTorchCost
from decent_bench.metrics import metric_library as ml
from decent_bench.metrics import runtime_library
from decent_bench.metrics.metric_utils import single
from decent_bench.networks import Network, P2PNetwork
from decent_bench.schemes import UniformActivationRate, UniformDropRate
from decent_bench.utils.checkpoint_manager import CheckpointManager
from decent_bench.utils.types import SupportedDevices


def model_generator() -> torch.nn.Module:
    """Generate a model for the NIM dataset."""
    return NimModel(
        input_size=2,
        hidden_sizes=[256, 64, 64],
        output_size=1,
    )


def create_heatmap_plots(image_file: str | Path, result: benchmark.BenchmarkResult, save_path: Path) -> None:  # noqa: D103
    def heatmap_plot(
        network: Network,
        width: int,
        height: int,
        norm: float,
    ) -> list[list[float]]:
        xs = np.arange(0, width) / norm
        ys = np.arange(0, height) / norm
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        nx, ny = X.shape
        points = np.column_stack((X.ravel(), Y.ravel()))
        mats = []
        for agent in network.agents():
            if not isinstance(agent.cost, PyTorchCost):
                continue
            agent_activation = agent.cost.final_activation
            agent.cost.final_activation = nn.Sigmoid()
            out = agent.cost.predict(agent.x, torch.tensor(points, dtype=torch.float32))
            agent.cost.final_activation = agent_activation
            out_np = np.array(out)

            mats.append(out_np.reshape((nx, ny)))

        return mats

    image = Image.open(image_file).convert("L")
    image_array = np.array(image)
    height, width = image_array.shape
    feature_norm = max(height, width)

    for alg, networks in result.states.items():
        for j, n in enumerate(networks):
            heatmaps = heatmap_plot(n, width, height, feature_norm)
            fig, ax = plt.subplots(1, 5, figsize=(15, 10))
            fig.suptitle(f"{alg.name} - Trial {j + 1}")
            for i, hm in enumerate(heatmaps):
                ax[i].imshow(image_array, cmap="gray")
                ax[i].imshow(hm, cmap="hot", interpolation="nearest", vmin=0, vmax=1, alpha=0.8)
                ax[i].invert_yaxis()
                ax[i].axis("off")
                ax[i].set_title(f"Agent {i + 1}")
            plt.tight_layout()
            plt.savefig(save_path / f"heatmaps_{alg.name}_{j + 1}.png")
            plt.close()


if __name__ == "__main__":
    folder = Path("results/nim")
    iterations = 20_000
    samples_per_partition = None  # Set to a ~25'000 if you are running out of memory
    state_snapshot_period = 500
    test_samples = 50_000
    leakage = 0.0
    label_balance = 2.0
    image_file = "data/kth_floorplan_sample.png"
    batch_size = 512
    local_steps = [5, 10]
    device = SupportedDevices.GPU

    table_metrics = [
        ml.ConsensusError([min, np.average, max]),
        ml.GradientCalls([np.average, sum]),
        ml.SentMessages([np.average, sum]),
        ml.ReceivedMessages([np.average, sum]),
        ml.SentMessagesDropped([np.average, sum]),
        ml.GradientNorm([single]),
        ml.MSE([min, np.average, max]),
        ml.Loss([min, np.average, max]),
    ]

    plot_metrics = [
        [ml.ConsensusError([], x_log=False, y_log=True)],
        [ml.MSE([], x_log=False, y_log=True)],
        [ml.Loss([], x_log=False, y_log=False)],
    ]

    for n_agents, n_neighbors in [(5, 4), (5, 2)]:
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
        test_data = train_dataset.get_test_set(label_balance=1.0, num_samples=test_samples)
        for drops, activity in [
            (None, None),
            (True, None),
            (None, True),
            (True, True),
        ]:
            for alg in [
                "DGD",
                "KGT",
                "LED",
                "ProxSkip",
                "LT-ADMM",
                "LT-ADMM-EMA",
            ]:
                resume_benchmark = False
                checkpoint_path = folder / f"{n_agents}_{n_neighbors}" / f"test_{drops}_{activity}" / alg
                if checkpoint_path.exists():
                    resume_benchmark = True

                cm = CheckpointManager(
                    checkpoint_dir=checkpoint_path,
                    checkpoint_step=None,
                    keep_n_checkpoints=1,
                    benchmark_metadata={
                        "dataset": "NIM",
                        "n_agents": n_agents,
                        "n_neighbors": n_neighbors,
                        "drops": drops,
                        "activity": activity,
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
                        # If you want to use a threshold-based activation instead of sigmoid, uncomment this line
                        # final_activation=FinalActivation(threshold=0.5),
                        final_activation=nn.Sigmoid(),
                        batch_size=batch_size,
                        max_batch_size=batch_size,
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
                    message_drop=UniformDropRate(0.2) if drops else None,
                )
                problem = benchmark.BenchmarkProblem(
                    network=network,
                    test_data=test_data,
                )
                x0 = pytorch_initialization(network, all_same=True)
                if alg == "DGD":
                    algorithms = [
                        dec_algorithms.DGD(
                            step_size=0.1,
                            aux_step_size=1.0,
                            iterations=iterations,
                            x0=x0,
                        )
                    ]
                elif alg == "KGT":
                    algorithms = [
                        dec_algorithms.KGT(
                            iterations=iterations,
                            local_steps=ls,
                            step_size=0.025,
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
                            step_size=0.05,
                            aux_step_size=0.05,
                            comm_probability=1 / ls,
                            chi=1.0,
                            x0=x0,
                            name=f"ProxSkip (comm_prob=1/{ls})",
                        )
                        for ls in local_steps
                    ]
                elif alg == "LED":
                    algorithms = [
                        dec_algorithms.LED(
                            iterations=iterations,
                            local_steps=ls,
                            step_size=0.025,
                            aux_step_size=0.025,
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
                            step_size=0.025,
                            aux_step_size=0.025,
                            penalty=1.0,
                            x0=x0,
                            name=f"LT-ADMM (ls={ls})",
                        )
                        for ls in local_steps
                    ]
                elif alg == "LT-ADMM-EMA":
                    algorithms = [
                        LT_ADMM_EMA(
                            iterations=iterations,
                            local_steps=ls,
                            step_size=0.025,
                            aux_step_size=0.025,
                            penalty=1.0,
                            ema_factor=0.8,
                            send_ema_x=False,
                            use_z_ema=False,
                            x0=x0,
                            name=f"LT-ADMM-EMA (ls={ls})",
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
                        )
                    else:
                        result = benchmark.benchmark(
                            algorithms=algorithms,
                            benchmark_problem=problem,
                            n_trials=5 if alg == "ProxSkip" or any((drops, activity)) else 1,
                            show_speed=True,
                            show_trial=True,
                            checkpoint_manager=cm,
                            runtime_metrics=[runtime_library.RuntimeLoss(250)],
                        )
                else:
                    result = cm.load_benchmark_result()

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

                create_heatmap_plots(
                    image_file=image_file,
                    result=result,
                    save_path=cm.get_results_path(),
                )

                del result, metric_result, costs, agents, network, problem, algorithms
                # Garbage collection to free up memory before the next benchmark
                gc.collect()
