import math

import matplotlib.pyplot as plt
from src.dataset import NIMDatasetHandler
from src.tools.visualize_dataset import visualize_nim_dataset

if __name__ == "__main__":
    # NIM path-based LIDAR sampling usage:
    import torch

    nim_data = NIMDatasetHandler(
        image_file="examples/nim/data/kth_floorplan.png",
        n_partitions=2,
        samples_per_partition=2000000,
        seed=0,
        paths="examples/nim/data/kth_2_path.json",
        samples_per_pose=5,
        num_beams=5,
        fov=math.pi,
        max_range=100,
        scan_spacing=5.0,
        transform=torch.tensor,
        label_transform=torch.tensor,
    )

    anim = visualize_nim_dataset(
        nim_data,
        animate=False,
        path_index=[0, 1],
        fps=2,
        save_path=f"test_1-2.png",
    )

    print(
        nim_data.get_partitions()[0][0],
        type(nim_data.get_partitions()[0][0]),
    )
    print(len(nim_data.get_partitions()[0]))

    # plt.show()

    # NIM random sampling usage:
    # import torch

    # path = "data/NIM/test.png"
    # nim_data = NIMData(
    #     image_file=path,
    #     partitions=1,
    #     leakage=0.2,
    #     samples_per_partition=500,
    #     seed=0,
    #     transform=torch.tensor,
    #     label_transform=torch.tensor,
    #     balance_labels=True,
    # )
    # fig = _visualize_nim_partitions(nim_data)

    # print(
    #     nim_data.training_partitions[0][0],
    #     type(nim_data.training_partitions[0][0]),
    # )

    # # print counts of each class in each partition
    # for i, partition in enumerate(nim_data.training_partitions):
    #     labels = np.array([item[1] for item in partition])
    #     unique, counts = np.unique(labels, return_counts=True)
    #     class_counts = dict(zip(unique, counts))
    #     print(f"Partition {i}: Class distribution: {class_counts}")

    # plt.show()

    # MNIST example

    # from torchvision.datasets import MNIST
    # from torchvision import transforms

    # transform = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))]
    # )
    # mnist_train = MNIST(root="./data", train=True, download=True, transform=transform)
    # mnist_train_wrapped = PyTorchWrapper(
    #     torch_dataset=mnist_train,
    #     classes=len(mnist_train.classes),
    #     features=784,  # 28*28 images
    #     partitions=10,
    #     # samples_per_partition=25 * num_agents,  # 60000 total samples
    #     seed=42,
    # )

    # for d in mnist_train_wrapped.training_partitions:
    #     for x, y in d:
    #         print(y)
