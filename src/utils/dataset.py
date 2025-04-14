import torch
from torch.utils.data import Subset, random_split
from torchvision import datasets, transforms


def load_mnist_data(
    training_size: int,
    validation_size: int,
    test_size: int,
) -> tuple[Subset, Subset, Subset]:

    # Transform for normalization
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # Mean and std of MNIST
        ]
    )

    # Load complete dataset
    full_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # # get full dataset
    # training_size = int(len(full_dataset) * (3 / 4))
    # validation_size = int(len(full_dataset) * (1 / 4))
    # test_size = len(test_dataset)

    # Select a subset of examples from the full training dataset
    subset_indices = torch.randperm(len(full_dataset))[: training_size + validation_size]
    dataset_subset = torch.utils.data.Subset(full_dataset, subset_indices)

    # Split the subset into training and validation sets
    train_dataset, val_dataset = random_split(dataset_subset, [training_size, validation_size])

    # Select a random subset for the test set
    test_indices = torch.randperm(len(test_dataset))[:test_size]
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_subset)}")

    return train_dataset, val_dataset, test_subset
