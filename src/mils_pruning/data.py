import os
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np



def load_mnist_dataset(img_size=(10, 10), data_root=None):
    if data_root is None:
        # This resolves to the project root *assuming you run from the notebooks folder*
        data_root = os.path.abspath(os.path.join(os.getcwd(), "../data"))

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(root=data_root, train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root=data_root, train=False, transform=transform, download=True)

    return train_dataset, test_dataset


def split_train_val(train_dataset, val_size=10000, random_state=42):
    targets = train_dataset.targets.numpy()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    for train_idx, val_idx in sss.split(np.zeros(len(targets)), targets):
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)
        return train_subset, val_subset, train_idx


def create_stratified_samples(train_dataset, train_indices, num_runs=200):
    full_targets = train_dataset.targets[train_indices].numpy()
    unique_classes = np.unique(full_targets)
    sampled_datasets = []

    for _ in range(num_runs):
        stratified_indices = []
        for cls in unique_classes:
            cls_idx = np.where(full_targets == cls)[0]
            n_sample = len(cls_idx) // 2
            sampled_cls_idx = np.random.choice(cls_idx, size=n_sample, replace=False)
            stratified_indices.extend(sampled_cls_idx)

        sampled_indices = np.array(train_indices)[stratified_indices]
        sampled_datasets.append(Subset(train_dataset, sampled_indices))

    return sampled_datasets


def get_mnist_data_loaders(batch_size=128, val_size=10000, num_runs=200, img_size=(10, 10), data_root="./data"):
    full_train_dataset, test_dataset = load_mnist_dataset(img_size, data_root)
    train_subset, val_subset, train_indices = split_train_val(full_train_dataset, val_size)
    sampled_datasets = create_stratified_samples(full_train_dataset, train_indices, num_runs)

    train_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in sampled_datasets]
    val_loader = DataLoader(val_subset, batch_size=10000, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False)

    return train_loaders, val_loader, test_loader


if __name__ == "__main__":
    train_loaders, val_loader, test_loader = get_mnist_data_loaders()
    print(f"Validation subset size: {len(val_loader.dataset)}")
    print(f"Generated {len(train_loaders)} stratified training subsets, each of size {len(train_loaders[0].dataset)}.")
