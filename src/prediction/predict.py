import os
import sys
from typing import Tuple, Dict

import structlog
import argparse
import torch
from tabulate import tabulate

logger = structlog.get_logger()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the project root to the sys.path
sys.path.insert(0, PROJECT_ROOT)

from src.training.train_model import WeatherClassifier, DEFAULT_DATASET_PATH
from src.utils.data_loader import WeatherDataset
from src.evaluation.evaluation import MODEL_PATH

DEFAULT_IMAGE_PATH = os.path.join(PROJECT_ROOT, "test_image")


def arguments() -> Tuple[str, str, str, str]:
    """
    Parses command-line arguments for the weather classification prediction script.

    Returns:
        tuple: A tuple containing:
               - model_path (str): Path to the trained model file.
               - images_path (str): Path to the directory containing images to predict.
               - dataset_path (str): Path to the dataset directory.
               - device (str): The computation device to use ('cpu' or 'cuda').
    """
    parser = argparse.ArgumentParser(description='Predict weather class for own images.')

    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Path to the trained model.')
    parser.add_argument('--images', type=str, default=DEFAULT_IMAGE_PATH, help='Path to the images to predict.')
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET_PATH, help='Path to the dataset.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for prediction.',
                        choices=["cpu", "cuda", "mps"])

    args = parser.parse_args()

    return args.model, args.images, args.dataset, args.device


def predict(model_path: str, images_path: str, dataset_path: str, compute_device: str) -> Dict[str, str]:
    """
    Loads a trained model and performs predictions on a set of images.

    Args:
        model_path (str): Path to the trained PyTorch model file.
        images_path (str): Path to the directory containing images for prediction.
        dataset_path (str): Path to the dataset directory used to initialize the dataset structure.
        compute_device (str): The device to perform computation on e.g. 'cpu' or 'cuda'.

    Returns:
        dict: A dictionary where keys are image filenames and values are the predicted weather conditions.
    """
    predictions = dict()
    compute_device = torch.device(compute_device)

    # Get the number of classes from the (training)-dataset
    n_classes = len(os.listdir(dataset_path))

    # Load the model
    model = WeatherClassifier(num_classes=n_classes, compute_device=compute_device, verbosity=False)
    model.load_state_dict(torch.load(model_path))

    # Get the prediction dataset
    weather_dataset = WeatherDataset(data_folder=dataset_path)
    data_loader = weather_dataset.custom_image_loader(images_path)
    logger.info(f"Number of images found in in images folder: {len(data_loader.dataset)}\n")

    # Perform predictions
    for i, (image, image_path) in enumerate(data_loader):
        image_filename = data_loader.dataset.samples[i][0]
        image_filename = image_filename.split("\\")[-1]
        image = image.to(device)
        output = model(image)
        _, predicted = torch.max(output, 1)
        predictions[image_filename] = weather_dataset.dataset.classes[predicted.item()]

    table = tabulate(predictions.items(), headers=['Image', 'Predicted Weather'], tablefmt='orgtbl')
    logger.info("Predictions:\n")
    print(table)

    print("\n=======FINISHED=======")

    # TODO: Maybe add matpotlib code to show the images and their predictions

    return predictions


if __name__ == "__main__":
    model_path, images_path, dataset_path, device = arguments()

    logger.info("Parameters:")
    print(f"Model path: {model_path}")
    print(f"Images path: {images_path}")
    print(f"Dataset path: {dataset_path}")
    print(f"Device: {device}\n")

    predict(model_path, images_path, dataset_path, device)
