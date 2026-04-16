from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from PIL import Image
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
from decent_bench.networks import Network, P2PNetwork
from decent_bench.utils.checkpoint_manager import CheckpointManager
from decent_bench.utils.logger import start_logger
from decent_bench.utils.pytorch_utils import SimpleLinearModel
from decent_bench.utils.types import SupportedDevices
from examples.nim.src.algorithms.lt_admm_ema import LT_ADMM_EMA
from examples.nim.src.algorithms.lt_admm_torch_optimizer import LT_ADMM_TORCH

start_logger()

save_path = "results/test_test"
iterations = 5_000
state_snapshot_period = 250
test_samples = 5_000
leakage = 0.0
label_balance = 2.0
image_file = "data/kth_floorplan.png"
batch_size = 512  # 10_000
device = SupportedDevices.CPU
opt_cls = torch.optim.Adam

table_metrics = [
    ml.ConsensusError([min, np.average, max]),
    # ml.GradientCalls([np.average, sum]),
    # ml.SentMessages([np.average, sum]),
    ml.MSE([min, np.average, max]),
    ml.Loss([min, np.average, max]),
]

plot_metrics = [
    # [ml.ConsensusError([], x_log=False, y_log=True)],
    # [ml.MSE([], x_log=False, y_log=True)],
    [ml.Loss([], x_log=False, y_log=False)],
]


class Model(nn.Module):  # noqa: D101
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int):
        super().__init__()

        self.layers = nn.ModuleList()
        prev_size = input_size

        # Hidden layers
        for hidden_size in hidden_sizes:
            self.layers.append(torch.nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size

        # Output layer
        self.layers.append(torch.nn.Linear(prev_size, output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Apply sine activation after the first layer to create a non-linear decision boundary,
        # then ReLU for the rest of the hidden layers
        x = torch.sin(self.layers[0](x))

        for layer in self.layers[1:-1]:
            x = torch.relu(layer(x))

        # No activation on the output layer since we'll be using BCEWithLogitsLoss which expects raw logits
        res: torch.Tensor = self.layers[-1](x)

        return res


def model_generator() -> torch.nn.Module:
    """Generate a simple linear model for the NIM dataset."""
    return Model(
        input_size=2,
        hidden_sizes=[256, 64, 64],
        output_size=1,
    )


class FinalActivation(nn.Module):  # noqa: D101
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return (self.sigmoid(x) > self.threshold).long()


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
    # points = np.random.default_rng().choice(points, size=10_000, replace=False)
    mats = []
    for agent in network.agents():
        agent_activation = agent.cost.final_activation
        agent.cost.final_activation = nn.Sigmoid()
        out = agent.cost.predict(agent.x, torch.tensor(points, dtype=torch.float32))
        agent.cost.final_activation = agent_activation
        out_np = np.array(out)

        mats.append(out_np.reshape((nx, ny)))

    return mats


iop.set_seed(47)
train_dataset = NIMDatasetHandler(
    image_file=image_file,
    n_partitions=5,
    transform=torch.tensor,  # type: ignore[arg-type]
    label_transform=lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(0),  # type: ignore[arg-type]
    label_balance=label_balance,
    leakage=leakage,
)

costs = [
    PyTorchCost(
        dataset=p,
        model=model_generator(),
        loss_fn=nn.BCEWithLogitsLoss(),
        final_activation=FinalActivation(threshold=0.5),
        batch_size=batch_size,
        max_batch_size=batch_size,
        device=device,
        load_dataset=True,
        use_dataloader=False,
    )
    for p in train_dataset.get_partitions()
]
agents = [
    Agent(
        i,
        cost,
        state_snapshot_period=state_snapshot_period,
    )
    for i, cost in enumerate(costs)
]
graph = nx.random_regular_graph(d=4, n=5, seed=iop.get_seed())
network = P2PNetwork(
    graph=graph,
    agents=agents,
)
problem = benchmark.BenchmarkProblem(
    network=network,
    test_data=train_dataset.get_test_set(label_balance=1.0, num_samples=test_samples),
)
x0 = pytorch_initialization(network, all_same=True)
algorithms = [
    # dec_algorithms.KGT(
    #     iterations=iterations,
    #     local_steps=10,
    #     step_size=0.1,
    #     aux_step_size=0.5,
    #     x0=x0,
    #     name="KGT (ss=0.1, aux_ss=0.5)",
    # ),
    # dec_algorithms.LT_ADMM(
    #     iterations=iterations,
    #     local_steps=10,
    #     step_size=0.05,
    #     aux_step_size=0.05,
    #     penalty=1.0,
    #     mask_z=False,
    #     x0=x0,
    #     name="LT-ADMM (ss=0.05)",
    # ),
    # LT_ADMM_EMA(
    #     iterations=iterations,
    #     local_steps=10,
    #     step_size=0.005,
    #     aux_step_size=0.01,
    #     penalty=1.0,
    #     ema_factor=0.8,
    #     send_ema_x=False,
    #     use_z_ema=False,
    #     mask_z=False,
    #     x0=x0,
    #     name="LT-ADMM-EMA (ss=0.005)",
    # ),
    # LT_ADMM_EMA(
    #     iterations=iterations,
    #     local_steps=10,
    #     step_size=0.01,
    #     aux_step_size=0.01,
    #     penalty=1.0,
    #     ema_factor=0.8,
    #     send_ema_x=False,
    #     use_z_ema=False,
    #     mask_z=False,
    #     x0=x0,
    #     name="LT-ADMM-EMA (ss=0.01)",
    # ),
    # dec_algorithms.DGD(
    #     iterations=iterations,
    #     step_size=0.01,
    #     x0=x0,
    #     name="DGD (ss=0.01)",
    # ),
    # dec_algorithms.DGD(
    #     iterations=iterations,
    #     step_size=0.05,
    #     x0=x0,
    #     name="DGD (ss=0.05)",
    # ),
    dec_algorithms.LED(
        iterations=iterations,
        local_steps=10,
        step_size=0.01,
        aux_step_size=0.01,
        x0=x0,
        name="LED ss=0.01",
    )
]

for i in range(len(algorithms)):
    result = benchmark.benchmark(
        algorithms=algorithms[i : i + 1],
        benchmark_problem=problem,
        n_trials=1,
        show_speed=True,
        show_trial=True,
        # runtime_metrics=[runtime_library.RuntimeLoss(250)],
    )

    metric_result = benchmark.compute_metrics(
        benchmark_result=result,
        table_metrics=table_metrics,
        plot_metrics=plot_metrics,
    )

    benchmark.display_metrics(
        metrics_result=metric_result,
        show_plots=False,
        save_path=save_path + f"/{algorithms[i].name}",
    )

    image = Image.open(image_file).convert("L")
    image_array = np.array(image)
    height, width = image_array.shape
    feature_norm = max(height, width)

    for alg, networks in result.states.items():
        print(f"Results for {alg.name}, {len(networks)} trials")
        for j, n in enumerate(networks):
            heatmaps = heatmap_plot(n, width, height, feature_norm)
            fig, ax = plt.subplots(1, 5, figsize=(15, 10))
            fig.suptitle(f"Heatmaps for {alg.name}")
            for i, map in enumerate(heatmaps):
                ax[i].imshow(image_array, cmap="gray")
                ax[i].imshow(map, cmap="hot", interpolation="nearest", vmin=0, vmax=1, alpha=0.8)
                ax[i].invert_yaxis()
                ax[i].axis("off")
                ax[i].set_title(f"Heatmap for Agent {i + 1}")
            plt.tight_layout()
            plt.savefig(save_path + f"/heatmaps_{alg.name}_{j + 1}.png")
            plt.close()
