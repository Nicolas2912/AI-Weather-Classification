import argparse
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from tabulate import tabulate
from tqdm import tqdm

from train.train_model import WeatherClassifier
from utils.data_loader import WeatherDataset
import structlog

logger = structlog.get_logger()


def arguments():
    """
    Parses command line arguments for evaluating the weather classification model.

    Returns:
        Tuple[str, str, str]: A tuple containing:
        - data (str): Path to the dataset folder.
        - model (str): Path to the model file.
        - device (str): Device type (e.g., 'cpu' or 'cuda') to be used for training.
    """
    parser = argparse.ArgumentParser(description='Evaluate the weather classification model.')

    parser.add_argument('--data', type=str,
                        default="/Users/nicolasschneider/MeineDokumente/FH_Bielefeld/Optimierung_und_Simulation/2. "
                                "Semester/SimulationOptischerSysteme/AI-Weather-Classification/dataset",
                        help="Path to the dataset folder.")
    parser.add_argument('--model',
                        default="/Users/nicolasschneider/MeineDokumente/FH_Bielefeld/Optimierung_und_Simulation/2. "
                                "Semester/SimulationOptischerSysteme/AI-Weather-Classification/models/"
                                "trained_model_new.pth",
                        type=str, help="Path to the model file.")
    parser.add_argument('--device', type=str, default='cpu', help="Device to use for training.")

    args = parser.parse_args()

    dataset_path = args.data
    model_path = args.model
    device_type = args.device

    return dataset_path, model_path, device_type


def _plot_confusion_matrix(matrix: np.ndarray, class_names: List[str]):
    """
    Plots a confusion matrix using matplotlib.

    Args:
        matrix (np.ndarray): A 2D array of the confusion matrix where each element represents the count of predictions.
        class_names (List[str]): List of class names corresponding to the indices of the confusion matrix.

    Displays:
        A matplotlib figure visualizing the confusion matrix with class names and values.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(matrix, cmap="YlOrRd")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=13)
    ax.set_yticklabels(class_names, fontsize=13)
    ax.set_xlabel("Predicted label", fontsize=15)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel("True label", fontsize=15)
    ax.set_title("Confusion matrix", fontsize=16)
    ax.set_ylim(len(class_names) - 0.5, -0.5)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black", fontsize=15)
    plt.show()


def eval_model(classifier_model: WeatherClassifier, test_data: DataLoader, class_names: List[str]) -> Tuple[float, str]:
    """
    Evaluates the given model on the specified test data.

    This function performs evaluation by predicting over the test dataset, calculating the F1 score,
    precision, recall, and generating a detailed classification report and confusion matrix plot.

    Args:
        classifier_model (str): The pre-trained model to evaluate.
        test_data (DataLoader): DataLoader containing the test dataset.
        class_names (List[str]): List of names corresponding to the output classes.

    Returns:
        Tuple[float, str]:
        - F1 score of the model across all classes, weighted by support.
        - String representation of the full classification report.

    Effects:
        Prints out the F1 score, recall, and precision in a tabulated format and plots the confusion matrix.
    """
    print("=" * 100)
    logger.info("START EVALUATING MODEL ON TEST DATA")
    classifier_model.eval()  # Set the model to evaluation mode
    true_labels = []
    pred_labels = []

    with torch.no_grad():  # Deactivate gradients for the following code
        for images, labels in tqdm(test_data, desc="Evaluating model on test data"):
            images = images
            labels = labels
            outputs = classifier_model(images)  # Forward pass through the model
            _, preds = torch.max(outputs, 1)  # Get the index of the max log-probability

            true_labels.extend(labels.cpu().numpy())  # Move labels back to CPU and convert to numpy
            pred_labels.extend(preds.cpu().numpy())  # Move predictions back to CPU and convert to numpy

    f_1score = f1_score(true_labels, pred_labels, average="weighted")

    from sklearn.metrics import precision_score, recall_score

    recall_score = recall_score(true_labels, pred_labels, average="weighted")
    precision_score = precision_score(true_labels, pred_labels, average="weighted")

    round_decimals = 4

    # make table with accuracy and f1 score
    table = [["F1-Score", round(f_1score, round_decimals)], ["Recall", round(recall_score, round_decimals)],
             ["Precision", round(precision_score, round_decimals)]]
    print(tabulate(table, headers=["Metric", "Value"], tablefmt="pretty"))
    print()

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(true_labels, pred_labels)

    # plot confusion matrix
    _plot_confusion_matrix(conf_matrix, class_names)

    precision_data = classification_report(true_labels, pred_labels, target_names=class_names, output_dict=True)
    # round every value in the dictionary to 4 decimal places
    for key, value in precision_data.items():
        if isinstance(value, dict):
            precision_data[key] = {k: round(v, round_decimals) for k, v in value.items()}
        else:
            precision_data[key] = round(value, round_decimals)

    # Preprocess the dictionary to create a list of lists for tabulate
    table_data = []
    headers = ["Category", "Precision", "Recall", "F1-Score", "Support"]

    for category, metrics in precision_data.items():
        if isinstance(metrics, dict):  # Check if the value is a dictionary
            row = [category] + [metrics.get(key, '') for key in ['precision', 'recall', 'f1-score', 'support']]
            table_data.append(row)

    # Print the table using tabulate
    print(tabulate(table_data, headers=headers, tablefmt='pretty'))

    # Print classification report
    return f1_score(true_labels, pred_labels, average="weighted"), classification_report(true_labels, pred_labels,
                                                                                         target_names=class_names)


if __name__ == "__main__":
    # Get arguments
    data, model, device = arguments()

    # Load model
    state_dict = torch.load(model)

    W = WeatherDataset(data_folder=data)
    classes = W.classes

    # Initialize model
    model = WeatherClassifier(num_classes=len(classes), compute_device=device, verbosity=False)

    # Load model weights
    model.load_state_dict(state_dict)

    # Get test data
    tr, val, test = W.get_data_loaders(batch_size=1)

    f1_score, class_report = eval_model(model, test, classes)

    print("=======FINISHED=======")
