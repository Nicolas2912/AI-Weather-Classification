import os
import random

import torch

import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize, ToTensor
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from collections import defaultdict
from typing import Tuple, Dict

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class WeatherDataset(Dataset):
    """A dataset class for loading weather images with transformations.

    Attributes:
        data_folder (str): Path to the dataset folder.
        transform (callable, optional): Optional transform to be applied on a sample.

    Methods:
        __len__: Returns the total number of images in the dataset.
        __getitem__: Retrieves an image and its label by index.
        get_data_loaders: Splits the dataset into training, validation, and test sets and returns corresponding loaders.
    """

    def __init__(self, data_folder: str, transform=None, resize_format: tuple = (512, 512)):
        self.data_folder = data_folder
        self.resize_format = resize_format
        if transform is None:
            transform = transforms.Compose([Resize(self.resize_format), ToTensor()])
        self.transform = transform
        self.dataset = ImageFolder(self.data_folder, transform=self.transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label

    def get_data_loaders(self, batch_size=32, train_ratio=0.7, val_ratio=0.15, seed=42) -> Tuple[
        DataLoader, DataLoader, DataLoader]:
        """Creates and returns data loaders for the dataset after splitting into train, validation, and test sets.

        Args:
            batch_size (int): Number of samples in each batch.
            train_ratio (float): Fraction of the dataset to be used as training data.
            val_ratio (float): Fraction of the dataset to be used as validation data.
            seed (int): Seed for random operations to ensure reproducibility.

        Returns:
            tuple: Three DataLoader instances for training, validation, and test datasets.
        """
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        total_length = len(self)
        train_length = int(train_ratio * total_length)
        val_length = int(val_ratio * total_length)
        test_length = total_length - train_length - val_length

        train_dataset, val_dataset, test_dataset = random_split(
            self, [train_length, val_length, test_length],
            generator=torch.Generator().manual_seed(seed)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        return train_loader, val_loader, test_loader

    def one_image_loader(self, image_path: str):
        """Loads a single image from the dataset and returns a DataLoader instance.

        Args:
            image_path (str): Path to the image file.

        Returns:
            DataLoader: DataLoader instance containing the image.
        """
        image = ImageFolder(image_path, transform=self.transform)
        image_loader = DataLoader(image, batch_size=1, shuffle=False)
        return image_loader


def analyze_class_distribution(loaders: dict, idx_to_class: dict) -> Dict[str, Dict[str, int]]:
    """
    Analyzes the class distribution across training, validation, and test datasets.

    Args:
    - loaders (dict): Dictionary of DataLoaders.
    - idx_to_class (dict): Dictionary mapping label indices to class names.

    Returns:
    - distribution (dict): Dictionary containing class distributions for train, val, and test sets.
    """
    distribution = {'train': defaultdict(int), 'val': defaultdict(int), 'test': defaultdict(int)}

    for phase in ['train', 'val', 'test']:
        loader = loaders[phase]
        for images, labels in tqdm(loader, desc=f"Analyzing {phase} distribution"):
            for label in labels:
                class_name = idx_to_class[label.item()]
                distribution[phase][class_name] += 1

    return distribution


def plot_class_distributions(distribution: Dict[str, Dict[str, int]]):
    """
    Plots pie charts of class distributions for training, validation, and test sets.

    Args:
    - distribution (dict): Dictionary containing class distributions for train, val, and test sets.
    """
    # Define the number of subplots based on the distribution keys
    n = len(distribution)
    fig, axs = plt.subplots(1, n, figsize=(n * 5, 6))  # Adjust figsize as needed

    if n == 1:  # If only one phase is provided, wrap axs in a list
        axs = [axs]

    for ax, (phase, counts) in zip(axs, distribution.items()):
        # Prepare data for the pie chart
        labels = counts.keys()
        sizes = counts.values()
        total = sum(sizes)  # Total count for percentage calculation

        ax.pie(sizes, labels=labels, autopct=lambda p: '{:.1f}%'.format(p) if p > 0 else '', startangle=90)
        ax.set_title(f'{phase.capitalize()} Set Distribution (Total: {total})')
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    path_dataset = "/Users/nicolasschneider/MeineDokumente/FH_Bielefeld/Optimierung_und_Simulation/2. Semester/SimulationOptischerSysteme/AI-Weather-Classification/dataset"
    one_image = "/Users/nicolasschneider/MeineDokumente/FH_Bielefeld/Optimierung_und_Simulation/2. Semester/SimulationOptischerSysteme/AI-Weather-Classification/utils/one_image"

    path_dataset_win = r"C:\Users\Anwender\Desktop\Nicolas\Dokumente\FH Bielefeld\Optimierung und Simulation\2. Semester\SimulationOptischerSysteme\AI-Weather-Classification\dataset"
    one_image_win = r"C:\Users\Anwender\Desktop\Nicolas\Dokumente\FH Bielefeld\Optimierung und Simulation\2. Semester\SimulationOptischerSysteme\AI-Weather-Classification\test_image"

    W = WeatherDataset(data_folder=path_dataset_win)
    train_loader, val_loader, test_loader = W.get_data_loaders(batch_size=1, train_ratio=0.7, seed=42)
    print(type(train_loader))

    print(f"Number of images in the dataset: {len(W)}")
    print(f"Number of images in the training set: {len(train_loader.dataset)}")
    print(f"Number of images in the validation set: {len(val_loader.dataset)}")
    print(f"Number of images in the test set: {len(test_loader.dataset)}\n")

    class_to_idx = W.dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    loaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

    distribution = analyze_class_distribution(loaders, idx_to_class)
    plot_class_distributions(distribution)

    # Load a single image from the dataset
    image_loader = W.one_image_loader(one_image_win)
    image, _ = next(iter(image_loader))
    # Display the image and its label
    plt.imshow(image.squeeze().permute(1, 2, 0))
    plt.title(f"Custom image")
    plt.axis('off')
    plt.show()
