import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from models.train_model import WeatherClassifier
from utils.data_loader import WeatherDataset
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from tabulate import tabulate

ROOT_PATH = r"C:\Users\Anwender\Desktop\Nicolas\Dokumente\FH Bielefeld\Optimierung und Simulation\2. Semester\SimulationOptischerSysteme\AI-Weather-Classification\dataset"

class_names = sorted(os.listdir(ROOT_PATH))


def eval_model(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    true_labels = []
    pred_labels = []

    with torch.no_grad():  # Deactivate gradients for the following code
        for images, labels in dataloader:
            images = images.to(device)  # Move images to the device currently used
            labels = labels.to(device)  # Move labels to the device currently used
            outputs = model(images)  # Forward pass through the model
            _, preds = torch.max(outputs, 1)  # Get the index of the max log-probability

            true_labels.extend(labels.cpu().numpy())  # Move labels back to CPU and convert to numpy
            pred_labels.extend(preds.cpu().numpy())  # Move predictions back to CPU and convert to numpy

    # Print classification report
    return classification_report(true_labels, pred_labels, target_names=class_names)


def eval_model_matrix(model, dataloader, device, class_to_idx_mapping: dict):
    model.eval()  # Set the model to evaluation mode
    true_labels = []
    pred_labels = []

    with torch.no_grad():  # Deactivate gradients for the following code
        for images, labels in dataloader:
            images = images.to(device)  # Move images to the device currently used
            labels = labels.to(device)  # Move labels to the device currently used
            outputs = model(images)  # Forward pass through the model
            _, preds = torch.max(outputs, 1)  # Get the index of the max log-probability

            true_labels.extend(labels.cpu().numpy())  # Move labels back to CPU and convert to numpy
            pred_labels.extend(preds.cpu().numpy())  # Move predictions back to CPU and convert to numpy

    print(true_labels)
    print(pred_labels)

    idx_to_class_mapping = {value: key for key, value in class_to_idx_mapping.items()}

    print(idx_to_class_mapping)

    true_labels = [idx_to_class_mapping[i] for i in true_labels]
    pred_labels = [idx_to_class_mapping[i] for i in pred_labels]

    # true_labels = [dataloader.dataset.dataset.classes[i] for i in true_labels]
    # pred_labels = [dataloader.dataset.dataset.classes[i] for i in pred_labels]

    return true_labels, pred_labels


if __name__ == "__main__":
    # load model
    model_path = r"C:\Users\Anwender\Desktop\Nicolas\Dokumente\FH Bielefeld\Optimierung und Simulation\2. Semester\SimulationOptischerSysteme\AI-Weather-Classification\models"
    model_name = "trained_model.pth"

    model = WeatherClassifier(num_classes=11, device=torch.device("cpu"))
    state_dict = torch.load(os.path.join(model_path, model_name))
    model.load_state_dict(state_dict)

    # load data
    W = WeatherDataset(data_folder=ROOT_PATH)
    tr, val, test = W.get_data_loaders(batch_size=1)

    # evaluate model
    # set device to cpu
    device = torch.device("cpu")
    true_labels, pred_labels = eval_model_matrix(model, test, device, W.dataset.class_to_idx)
    print(true_labels)
    print(pred_labels)

    print(len(true_labels))
    print(len(pred_labels))

    # ConfusionMatrixDisplay.from_predictions(true_labels, pred_labels, display_labels=class_names)
