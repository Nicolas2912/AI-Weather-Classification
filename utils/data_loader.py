import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

import os


class WeatherDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        if transform is None:
            transform = transforms.Compose([Resize((256, 256)), transforms.ToTensor()])
        self.transform = transform
        self.dataset = ImageFolder(self.data_folder, transform=self.transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Load and preprocess a single sample
        image, label = self.dataset[index]
        return image, label

    def one_image_loader(self):
        image_loader = DataLoader(self.dataset, batch_size=1)
        return image_loader
        # test_image_loader = DataLoader()

    def get_data_loaders(self, batch_size=32, train_ratio=0.7, val_ratio=0.15):
        train_length = int(train_ratio * len(self))
        val_length = int(val_ratio * len(self))
        test_length = len(self) - train_length - val_length

        # split the dataset into train, validation and test
        train_dataset, val_dataset, test_dataset = random_split(self, [train_length, val_length, test_length])

        # create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        return train_loader, val_loader, test_loader

    def _plot_image_sizes(self):
        # Initialize two empty lists to store the heights and widths
        image_heights = []
        image_widths = []

        # Iterate over the dataset
        for i, (image, label) in tqdm(enumerate(W.dataset)):
            # Get the size of the image (height and width)
            _, height, width = image.shape
            image_heights.append(height)
            image_widths.append(width)

        # Calculate the average height and width
        avg_height = int(sum(image_heights) / len(image_heights))
        avg_width = int(sum(image_widths) / len(image_widths))

        print(f"Average image height: {avg_height}")
        print(f"Average image width: {avg_width}")

        # middle point
        print(f"Middle point: {(avg_height, avg_width)}")

        plt.scatter(image_widths, image_heights, alpha=0.1)
        plt.scatter(avg_width, avg_height, color='red')
        plt.xlabel("Image width")
        plt.ylabel("Image height")
        plt.show()


if __name__ == "__main__":
    path_dataset = "/Users/nicolasschneider/MeineDokumente/FH_Bielefeld/Optimierung_und_Simulation/2. Semester/SimulationOptischerSysteme/AI-Weather-Classification/dataset"
    one_image = "/Users/nicolasschneider/MeineDokumente/FH_Bielefeld/Optimierung_und_Simulation/2. Semester/SimulationOptischerSysteme/AI-Weather-Classification/utils/one_image"
    W = WeatherDataset(data_folder=one_image)
    image_loader = W.one_image_loader()
    print(image_loader.dataset)
    print(W.dataset.class_to_idx)
    # tr, val, test = W.get_data_loaders(batch_size=1)
    # for i, (image, label) in enumerate(tr):
    #     print(image.shape)
    #     print(label)
    #     break

